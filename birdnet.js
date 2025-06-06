class MelSpecLayerSimple extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.sampleRate = config.sampleRate
        this.specShape = config.specShape
        this.frameStep = config.frameStep
        this.frameLength = config.frameLength
        this.melFilterbank = tf.tensor2d(config.melFilterbank)
    }
    build(inputShape) {
        this.magScale = this.addWeight(
            'magnitude_scaling',
            [],
            'float32',
            tf.initializers.constant({ value: 1.23 })
        );
        super.build(inputShape)
    }
    computeOutputShape(inputShape) {
        return [inputShape[0], this.specShape[0], this.specShape[1], 1];

    }

    normalise_audio_batch = (tensor) => {
        return tf.tidy(() => {
            const sigMax = tf.max(tensor, 1, true);
            const sigMin = tf.min(tensor, 1, true);
            const range = sigMax.sub(sigMin);

            const two = tf.scalar(2);
            const one = tf.scalar(1);

            const normalized = tensor
            .sub(sigMin)
            .divNoNan(range)
            .mul(two)
            .sub(one);

            return normalized;
        });
    };
    call(inputs) {
    return tf.tidy(() => {
      // inputs is a tensor representing the input data
      inputs = inputs[0];
      let result;
      const BACKEND = tf.getBackend();
      if (BACKEND === 'blah') {
        result = tf.stack(
          inputs.split(inputs.shape[0]).map((input) => {
            input = input.squeeze();
            
            // Normalize values between -1 and 1
            input = this.normalise_audio_batch(input);
            // Perform STFT and cast result to float
            return tf.signal.stft(
              input,
              this.frameLength,
              this.frameStep,
              this.frameLength,
              tf.signal.hannWindow
            ).cast("float32");
          })
        )
      } else {
        // Normalise batch
        inputs = this.normalise_audio_batch(inputs);
        //Custom optimized and batch-capable stft
        result = stft(
          inputs,
          this.frameLength,
          this.frameStep,
          this.frameLength,
          tf.signal.hannWindow
        )
      }
      return result
        .matMul(this.melFilterbank) 
        .pow(2.0)
        .pow(tf.div(1.0, tf.add(1.0, tf.exp(this.magScale.read()))))
        .reverse(-1)
        .transpose([0, 2, 1])
        .expandDims(-1);
    });
  }
    static get className() { return 'MelSpecLayerSimple' }
}
tf.serialization.registerClass(MelSpecLayerSimple)

function stft(signal, frameLength, frameStep, fftLength, windowFn) {
    const framedSignal = tf.engine().runKernel('FRAME', {input: signal, frameLength, frameStep })
    const input = tf.mul(framedSignal, windowFn(frameLength))
    let innerDim = input.shape[input.shape.length - 1]
    const batch = input.size / innerDim
    const realValues = tf.engine().runKernel('FFT2', {input: tf.reshape(input, [batch, innerDim])})
    const half = Math.floor(innerDim / 2) + 1
    const realComplexConjugate = tf.split(
        realValues, [half, innerDim - half],
        realValues.shape.length - 1)
    const outputShape = input.shape.slice()
    outputShape[input.shape.length - 1] = half
    return tf.reshape(realComplexConjugate[0], outputShape)
}

tf.registerKernel({
    kernelName: 'FFT2',
    backendName: 'webgl',
    kernelFunc: ({ backend, inputs: { input } }) => {
        const innerDim = input.shape[input.shape.length - 1] / 2
        const batch = tf.util.sizeFromShape(input.shape) / innerDim / 2
        let currentTensor = backend.runWebGLProgram({
            variableNames: ['mapvalue'],
            outputShape: [batch, innerDim * 2],
            userCode: `
void main() {
  ivec2 coords = getOutputCoords();
  int p = coords[1] % ${innerDim};
  int k = 0;
  for (int i = 0; i < ${Math.log2(innerDim)}; ++i) {
    if ((p & (1 << i)) != 0) { k |= (1 << (${Math.log2(innerDim) - 1} - i)); }
  }
  if (coords[1] < ${innerDim}) {
    setOutput(getMapvalue(coords[0], 2 * k));
  } else {
    setOutput(getMapvalue(coords[0], 2 * (k % ${innerDim}) + 1));
  }
}`
        }, [input], 'float32')
        for (let len = 1; len < innerDim; len *= 2) {
            let prevTensor = currentTensor
            currentTensor = backend.runWebGLProgram({
                variableNames: [`x`],
                outputShape: [batch, innerDim * 2],
                userCode: `void main() {
ivec2 coords = getOutputCoords();
int batch = coords[0];
int i = coords[1];
int k = i % ${innerDim};
int isHigh = (k % ${len * 2}) / ${len};
int highSign = (1 - isHigh * 2);
int baseIndex = k - isHigh * ${len};
float t = ${Math.PI / len} * float(k % ${len});
float a = cos(t);
float b = sin(-t);
float oddK_re = getX(batch, baseIndex + ${len});
float oddK_im = getX(batch, baseIndex + ${len + innerDim});
if (i / ${innerDim} == 0) { // real
    float evenK_re = getX(batch, baseIndex);
    float outp = evenK_re + (oddK_re * a - oddK_im * b) * float(highSign);
    setOutput(outp);
} else { // imaginary
    float evenK_im = getX(batch, baseIndex + ${innerDim});
    float outp = evenK_im + (oddK_re * b + oddK_im * a) * float(highSign);
    setOutput(outp);
}
}` }, [currentTensor], 'float32')
            backend.disposeIntermediateTensorInfo(prevTensor)
        }

        let prevTensor = currentTensor
        currentTensor = backend.runWebGLProgram({
            variableNames: ['x'],
            outputShape: [batch, innerDim * 2],
            userCode: `
void main() {
    ivec2 coords = getOutputCoords();
    int i = coords[1];
    int batch = coords[0];

    int k = i <= ${innerDim} ? i : ${innerDim * 2} - i;
    int zI = k % ${innerDim};
    int conjI = (${innerDim} - k) % ${innerDim};
    float Zk0 = getX(batch, zI);
    float Zk_conj0 = getX(batch, conjI);
    float t = ${-2 * Math.PI} * float(k) / float(${innerDim * 2});
    float result = (Zk0 + Zk_conj0 + cos(t) * (getX(batch, zI+${innerDim}) + getX(batch, conjI+${innerDim})) + sin(t) * (Zk0 - Zk_conj0)) * 0.5;
    setOutput(result);
}`
        }, [currentTensor], 'float32')
        backend.disposeIntermediateTensorInfo(prevTensor)
        return currentTensor
    }
})
tf.registerKernel({
  kernelName: "FRAME",
  backendName: "webgl",
  kernelFunc: ({ inputs: { input, frameLength, frameStep }, backend }) => {
    const [batchSize, signalLength] = input.shape;
    const outputLength = (signalLength - frameLength + frameStep) / frameStep | 0;
    const outputShape = [batchSize, outputLength, frameLength];
    const userCode = `void main() {
        ivec3 coords = getOutputCoords(); // [batch, frame, sample]
        int b = coords.x;
        int f = coords.y;
        int l = coords.z;

        int signalIndex = f * ${frameStep} + l;
        float value = getX(b, signalIndex);
        setOutput(value);
      }`;
    return backend.compileAndRun({
        variableNames: ["x"],
        outputShape,
        userCode,
      }, [input]);
  },
});
function arrayProduct (arr) {
    let product = 1;
    for (let i = 0; i < arr.length; i++) { product *= arr[i] }
    return product;
}
function flatDispatchLayout(shape) { return {x: shape.map((d, i) => i)} }
function computeDispatch(layout, outputShape, workgroupSize = [1, 1, 1], elementsPerThread = [1, 1, 1]) {
return [Math.ceil(arrayProduct(layout.x.map(d => outputShape[d])) /(workgroupSize[0] * elementsPerThread[0])),
    layout.y ? Math.ceil(arrayProduct(layout.y.map(d => outputShape[d])) / (workgroupSize[1] * elementsPerThread[1])) : 1,
    layout.z ? Math.ceil(arrayProduct(layout.z.map(d => outputShape[d])) / (workgroupSize[2] * elementsPerThread[2])) : 1]
}

if (!globalThis.tf) { globalThis.tf = {} }
tf.registerKernel?.({
    kernelName: 'FFT2',
    backendName: 'webgpu',
    kernelFunc: ({ backend, inputs: { input } }) => {
        const innerDim = input.shape[input.shape.length - 1] / 2
        const batch = tf.util.sizeFromShape(input.shape) / innerDim / 2
        const workgroupSize = [64, 1, 1]
        const dispatchLayout = flatDispatchLayout([batch, innerDim * 2])
        const dispatch = computeDispatch(dispatchLayout, [batch, innerDim * 2], workgroupSize, [2, 1, 1])
        let currentTensor = backend.runWebGPUProgram({
            variableNames: ['X'],
            outputShape: [batch, innerDim * 2],
            workgroupSize,
            shaderKey: `fft_permut_${innerDim}`,
            dispatchLayout,
            dispatch,
            getUserCode: () => `
fn main(index: i32) {
let batch = index / ${innerDim};
let p = index % ${innerDim};
let outIndexReal = batch * ${innerDim * 2} + p;
let outIndexImag = outIndexReal + ${innerDim};
var k = 0;
for (var i: u32 = 0; i < ${Math.log2(innerDim)}; i = i + 1) {
    if ((p & (1 << i)) != 0) { k |= (1 << (${Math.log2(innerDim) - 1} - i)); }
}
setOutputAtIndex(outIndexReal, getX(batch, 2 * k));
setOutputAtIndex(outIndexImag, getX(batch, 2 * (k % ${innerDim}) + 1));
}`
        }, [input], 'float32')
        for (let len = 1; len < innerDim; len *= 2) {
            let prevTensor = currentTensor
            currentTensor = backend.runWebGPUProgram({
                variableNames: [`value`],
                outputShape: [batch, innerDim * 2],
                workgroupSize,
                shaderKey: `fft_step_${innerDim}_${len}`,
                dispatchLayout,
                dispatch,
                getUserCode: () => `fn main(index: i32) {
                    let batch = index / ${innerDim};
                    let i = index % ${innerDim};
                    let outIndexReal = batch * ${innerDim * 2} + i;
                    let outIndexImag = outIndexReal + ${innerDim};
                    let k = i % ${innerDim};
                    let isHigh = (k % (${len} * 2)) / ${len};
                    let highSign = (1 - isHigh * 2);
                    let baseIndex = k - isHigh * ${len};
                    let t = ${Math.PI} / f32(${len}) * f32(k % ${len});
                    let a = cos(t);
                    let b = sin(-t);
                    let oddK_re = getValue(batch, baseIndex + ${len});
                    let oddK_im = getValue(batch, baseIndex + ${len} + ${innerDim});

                    let evenK_re = getValue(batch, baseIndex);
                    let outpR = (evenK_re + (oddK_re * a - oddK_im * b) * f32(highSign));
                    setOutputAtIndex(outIndexReal, outpR);
                    let evenK_im = getValue(batch, baseIndex + ${innerDim});
                    let outpI = (evenK_im + (oddK_re * b + oddK_im * a) * f32(highSign));
                    setOutputAtIndex(outIndexImag, outpI);
                    }`
                }, [currentTensor], 'float32')
            backend.disposeData(prevTensor.dataId)
        }
        let prevTensor = currentTensor
        currentTensor = backend.runWebGPUProgram({
            variableNames: ['x'],
            outputShape: [batch, innerDim * 2],
            workgroupSize,
            shaderKey: `fft_post_${innerDim}`,
            dispatchLayout,
            dispatch: computeDispatch(flatDispatchLayout([batch, innerDim * 2]), [batch, innerDim * 2], workgroupSize, [1, 1, 1]),
            getUserCode: () => `
fn main(index: i32) {
    let coords = getOutputCoords();
    let i = coords[1];
    let batch = coords[0];
    var k = i;
    if (i > ${innerDim}) {
      k = ${innerDim * 2} - i;
    }
    let zI = k % ${innerDim};
    let conjI = (${innerDim} - k) % ${innerDim};
    let Zk0 = getX(batch, zI);
    let Zk_conj0 = getX(batch, conjI);
    let t = ${-2 * Math.PI} * f32(k) / f32(${innerDim * 2});
    let result = (Zk0 + Zk_conj0 + cos(t) * (getX(batch, zI+${innerDim}) + getX(batch, conjI+${innerDim})) + sin(t) * (Zk0 - Zk_conj0)) * 0.5;
    setOutputAtIndex(index, result);
}`
        }, [currentTensor], 'float32')
        backend.disposeData(prevTensor.dataId)
        return currentTensor
    }
})
tf.registerKernel({
  kernelName: "FRAME",
  backendName: "webgpu",
  kernelFunc: ({ backend, inputs: { input, frameLength, frameStep } }) => {
    const [batchSize, signalLength] = input.shape;
    const outputLength = (signalLength - frameLength + frameStep) / frameStep | 0;
    const outputShape = [batchSize, outputLength, frameLength];
    const workgroupSize = [64, 1, 1]; // tune as needed
    const dispatchLayout = flatDispatchLayout(outputShape);
    const dispatch = computeDispatch(dispatchLayout, outputShape, workgroupSize);
    return backend.runWebGPUProgram(
      {
        variableNames: ["x"],
        outputShape,
        workgroupSize,
        dispatchLayout,
        dispatch,
        shaderKey: `frame_batched_${frameLength}_${frameStep}`,
        getUserCode: () => `
          fn main( index: i32) {
            let globalId = getOutputCoords();
            let b = i32(globalId.x);  // batch index
            let f = i32(globalId.y);  // frame index
            let l = i32(globalId.z);  // position within frame

            let signalIndex = f * ${frameStep} + l;

            // Get the value from input[b, signalIndex]
            setOutputAtCoords(b, f, l, getX(b, signalIndex));
          }`
      }, [input], "float32");
  },
});