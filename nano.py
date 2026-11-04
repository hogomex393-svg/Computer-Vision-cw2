import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import numpy as np

from runtime import demo, latency_summary


def digest(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command_output(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=15)
        return {'exit_code': result.returncode, 'output': result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'unavailable', 'reason': str(error)}


def environment():
    result = {'platform': platform.platform(), 'machine': platform.machine(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'jetpack': command_output(['dpkg-query', '-W', 'nvidia-jetpack']),
              'power_mode': command_output(['nvpmodel', '-q']),
              'tegrastats': shutil.which('tegrastats') or 'unavailable'}
    try:
        import tensorrt as trt
        result['tensorrt'] = trt.__version__
    except ImportError:
        result['tensorrt'] = 'unavailable'
    result['target_matches'] = result['machine'] == 'aarch64' and result['tensorrt'].startswith('8.')
    return result


def build_engine(onnx_path, output, precision='fp16', workspace_mb=256):
    import tensorrt as trt
    if not trt.__version__.startswith('8.'):
        raise RuntimeError('This implementation targets TensorRT 8.x, found ' + trt.__version__)
    onnx_path, output = Path(onnx_path), Path(output)
    metadata = json.loads(onnx_path.with_suffix('.json').read_text())
    if digest(onnx_path) != metadata['onnx_sha256']:
        raise ValueError('ONNX checksum does not match export receipt')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError('Expected one image input and one logits output')
    shape = tuple(network.get_input(0).shape)
    if len(shape) != 4 or shape[:2] != (1, 3) or shape[2] != shape[3] or min(shape) <= 0:
        raise ValueError('Expected fixed batch=1 square RGB input')
    config = builder.create_builder_config()
    config.max_workspace_size = int(workspace_mb) * 1024 * 1024
    if precision == 'fp16':
        if not builder.platform_has_fast_fp16:
            raise RuntimeError('FP16 requested but platform does not report fast FP16')
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != 'fp32':
        raise ValueError('Only FP32/FP16 are supported')
    started = time.perf_counter()
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError('TensorRT engine build failed')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(engine.serialize()))
    record = {'onnx_sha256': metadata['onnx_sha256'], 'engine_sha256': digest(output),
              'bytes': output.stat().st_size, 'precision': precision, 'input_shape': list(shape),
              'build_seconds': time.perf_counter() - started, 'environment': environment(),
              'smoke': metadata['smoke'], 'normalization': metadata['normalization'], 'portable': False}
    output.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


class TensorRTRunner:
    def __init__(self, path, asynchronous=True):
        import tensorrt as trt
        import pycuda.driver as cuda
        if not trt.__version__.startswith('8.'):
            raise RuntimeError('TensorRT 8.x required')
        self.cuda, self.asynchronous = cuda, asynchronous
        self.metadata = json.loads(Path(path).with_suffix('.json').read_text())
        self.mean, self.std = self.metadata['normalization']['mean'], self.metadata['normalization']['std']
        if digest(path) != self.metadata['engine_sha256']:
            raise ValueError('Engine checksum mismatch')
        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()
        self.allocations = []
        self.hosts = []
        self.closed = False
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None or self.engine.has_implicit_batch_dimension:
                raise ValueError('Cannot load explicit-batch engine')
            if self.engine.num_bindings != 2:
                raise ValueError('Expected exactly two bindings')
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.bindings, self.shapes = [], []
            self.input_index = self.output_index = None
            dtype_map = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32}
            for i in range(self.engine.num_bindings):
                shape = tuple(self.engine.get_binding_shape(i))
                if min(shape) <= 0:
                    raise ValueError('Dynamic engine bindings are not supported')
                dtype = dtype_map[self.engine.get_binding_dtype(i)]
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                allocation = cuda.mem_alloc(host.nbytes)
                self.hosts.append(host)
                self.allocations.append(allocation)
                self.bindings.append(int(allocation))
                self.shapes.append(shape)
                if self.engine.binding_is_input(i):
                    self.input_index = i
                else:
                    self.output_index = i
            if self.input_index is None or self.output_index is None:
                raise ValueError('Input/output bindings missing')
            shape = self.shapes[self.input_index]
            if shape[0:2] != (1, 3) or shape[2] != shape[3]:
                raise ValueError('Invalid image binding shape')
            self.size = shape[-1]
            self.begin, self.end = cuda.Event(), cuda.Event()
        except Exception:
            for allocation in self.allocations:
                allocation.free()
            self.cuda_context.pop()
            self.cuda_context.detach()
            self.closed = True
            raise
        self.cuda_context.pop()

    def __call__(self, tensor):
        if self.closed:
            raise RuntimeError('Runner is closed')
        i, o = self.input_index, self.output_index
        if tensor.shape != self.shapes[i]:
            raise ValueError('Input tensor shape differs from engine')
        self.cuda_context.push()
        try:
            np.copyto(self.hosts[i], tensor.ravel(), casting='same_kind')
            if self.asynchronous:
                self.cuda.memcpy_htod_async(self.allocations[i], self.hosts[i], self.stream)
                self.begin.record(self.stream)
                ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                self.end.record(self.stream)
                self.cuda.memcpy_dtoh_async(self.hosts[o], self.allocations[o], self.stream)
                self.stream.synchronize()
            else:
                self.cuda.memcpy_htod(self.allocations[i], self.hosts[i])
                self.begin.record()
                ok = self.context.execute_v2(bindings=self.bindings)
                self.end.record()
                self.cuda.Context.synchronize()
                self.cuda.memcpy_dtoh(self.hosts[o], self.allocations[o])
            if not ok:
                raise RuntimeError('TensorRT execution returned failure')
            return self.hosts[o].reshape(self.shapes[o]).astype(np.float32).copy(), float(self.begin.time_till(self.end))
        finally:
            self.cuda_context.pop()

    def close(self):
        if self.closed:
            return
        self.cuda_context.push()
        try:
            self.stream.synchronize()
            for allocation in self.allocations:
                allocation.free()
            self.allocations = []
            self.hosts = []
            self.context = self.engine = self.runtime = None
            self.begin = self.end = self.stream = None
        finally:
            self.cuda_context.pop()
            self.cuda_context.detach()
            self.closed = True


def resource_summary(path):
    if not Path(path).exists():
        return {'status': 'unavailable'}
    text = Path(path).read_text(errors='replace')
    patterns = {'ram_used_mb': r'RAM (\d+)/', 'gpu_utilization_percent': r'GR3D_FREQ (\d+)%',
                'gpu_temperature_c': r'GPU@([\d.]+)C', 'cpu_temperature_c': r'CPU@([\d.]+)C',
                'power_input_mw': r'POM_5V_IN (\d+)'}
    result = {}
    for key, pattern in patterns.items():
        values = [float(value) for value in re.findall(pattern, text)]
        result[key] = {'mean': float(np.mean(values)), 'max': max(values), 'samples': len(values)} if values else 'unavailable'
    return result


def benchmark(engine, source, output, duration=600, asynchronous=False, smoke=False, loop=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    process = None
    log = None
    runner = TensorRTRunner(engine, asynchronous)
    if runner.metadata['smoke'] and not smoke:
        runner.close()
        raise ValueError('Smoke engine requires explicit --smoke')
    try:
        if shutil.which('tegrastats'):
            log = (output / 'tegrastats.log').open('w')
            process = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=log, stderr=subprocess.STDOUT)
        result = demo(runner, source, output, headless=True, asynchronous=asynchronous, warmup=30,
                      duration=duration, label=Path(engine).stem, smoke=smoke, loop=loop and not asynchronous)
        result['requested_duration_seconds'] = duration
        result['sustained_duration_met'] = result['wall_seconds'] >= duration
        result['environment'] = environment()
        result['engine'] = runner.metadata
    finally:
        runner.close()
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log:
            log.close()
    result['resources'] = resource_summary(output / 'tegrastats.log')
    (output / 'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['sustained_duration_met']:
        raise RuntimeError('Source ended before sustained benchmark duration; use a longer source or sync playback loop')
    return result


def verify(engine, reference, output, atol=0.05, mean_tolerance=0.005, mask_agreement=0.995):
    runner = TensorRTRunner(engine)
    samples = np.load(reference, allow_pickle=False)
    measurements = []
    try:
        for image, target in zip(samples['inputs'], samples['logits']):
            predicted, _ = runner(image)
            difference = np.abs(predicted - target)
            row = {'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()),
                   'mask_agreement': float((predicted.argmax(1) == target.argmax(1)).mean())}
            row['passed'] = row['max_abs'] <= atol and row['mean_abs'] <= mean_tolerance and row['mask_agreement'] >= mask_agreement
            measurements.append(row)
    finally:
        runner.close()
        samples.close()
    result = {'engine': str(engine), 'reference_sha256': digest(reference), 'measurements': measurements,
              'thresholds': {'max_abs': atol, 'mean_abs': mean_tolerance, 'mask_agreement': mask_agreement},
              'passed': bool(measurements) and all(row['passed'] for row in measurements)}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['passed']:
        raise RuntimeError('TensorRT/reference alignment failed; inspect ' + str(output))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['env', 'build', 'demo', 'benchmark', 'matrix', 'verify'])
    parser.add_argument('--onnx')
    parser.add_argument('--engine')
    parser.add_argument('--out', default='nano_results')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--workspace-mb', type=int, default=256)
    parser.add_argument('--source')
    parser.add_argument('--duration', type=float, default=600)
    parser.add_argument('--async-mode', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--models-dir')
    parser.add_argument('--model-prefix', default='S3')
    parser.add_argument('--reference')
    args = parser.parse_args()
    if args.command == 'env':
        result = environment()
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / 'environment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    elif args.command == 'build':
        if not args.onnx or not args.engine:
            parser.error('--onnx and --engine are required')
        result = build_engine(args.onnx, args.engine, args.precision, args.workspace_mb)
    elif args.command == 'verify':
        if not args.engine or not args.reference:
            parser.error('--engine and --reference are required')
        result = verify(args.engine, args.reference, Path(args.out) / 'alignment.json')
    elif args.command == 'matrix':
        if not args.models_dir or not args.source:
            parser.error('--models-dir and --source are required')
        result = {}
        for size in (256, 384, 512):
            for precision in ('fp32', 'fp16'):
                name = '{}_{}_{}'.format(args.model_prefix, size, precision)
                engine = Path(args.out) / 'engines' / (name + '.engine')
                build_engine(Path(args.models_dir) / ('{}_{}.onnx'.format(args.model_prefix, size)),
                             engine, precision, args.workspace_mb)
                verify(engine, Path(args.models_dir) / ('{}_{}.reference.npz'.format(args.model_prefix, size)),
                       Path(args.out) / name / 'alignment.json')
                result[name] = benchmark(engine, args.source, Path(args.out) / name, args.duration, args.async_mode, args.smoke)
        (Path(args.out) / 'matrix.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        if not args.engine or not args.source:
            parser.error('--engine and --source are required')
        if args.command == 'benchmark':
            result = benchmark(args.engine, args.source, args.out, args.duration, args.async_mode, args.smoke)
        else:
            runner = TensorRTRunner(args.engine, args.async_mode)
            try:
                if runner.metadata['smoke'] and not args.smoke:
                    raise ValueError('Smoke engine requires --smoke')
                result = demo(runner, args.source, args.out, not args.show, args.async_mode,
                              label=Path(args.engine).stem, smoke=args.smoke)
            finally:
                runner.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
