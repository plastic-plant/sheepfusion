// Runtime:   https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html
// Packages:  https://github.com/TensorStack-AI/OnnxStack/blob/master/OnnxStack.StableDiffusion/README.md

using Microsoft.Extensions.Logging;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

var pipeline = StableDiffusionPipeline.CreatePipeline(
    "model\\onnx",
    ModelType.Base,
    0,
    ExecutionProvider.Cpu,
    MemoryModeType.Maximum,
    null
);

var promptOptions = new PromptOptions { Prompt = "Photo of a cute sheepsnapshotmodel." };
var result = await pipeline.GenerateImageAsync(promptOptions);
await result.SaveAsync("output\\sheep001.png");
await pipeline.UnloadAsync();