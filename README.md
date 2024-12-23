CS330 - Operating Systems and Lab

### Project 4: Emerging Systems Project

# Latency Alleviation by Switching Between CPU and GPU for Parallel Artificial Intelligence Execution
**Minjae Jeong (20230653), Minjoon Jeong (20230654)**

## Introduction
The integration of AI models on mobile devices has enabled advanced features like real-time object detection and language processing. However, running multiple AI models simultaneously, such as a lightweight large language model (LLM) and an object detection model, often leads to computational challenges on resource-constrained devices. Mobile GPUs, while optimized for parallel tasks, are generally limited to handling a single inference process at a time. When multiple models compete for GPU resources, latency increases, resulting in degraded performance and potential system bottlenecks, which is especially problematic in scenarios requiring real-time responses.
This project addresses these challenges by implementing a dynamic resource allocation strategy that monitors inference latency and seamlessly switches the object detection task from GPU to CPU when latency exceeds a predefined threshold. This approach ensures real-time performance and optimal hardware utilization, even under heavy computational workloads. Through this work, we aim to explore the orchestration of CPU and GPU resources, measure performance differences, and propose scalable solutions for enhancing inference efficiency in mobile AI applications.

## Problem Solving
### What problem occurred? Why does it happen?
The problem in this scenario revolves around excessive inference latency when running object detection on a GPU, especially in a resource-constrained environment. Mobile devices often have limited computational capacity, and the GPU, while optimized for parallel computations, can become overwhelmed under certain conditions. Specifically, the issue occurs when the GPU inference time exceeds a threshold that impacts real-time performance. This could happen due to a variety of reasons, such as running multiple simultaneous AI models (e.g., a lightweight LLM and object detection), limited memory bandwidth, or the inherent complexity of the object detection model. Therefore, an adaptable approach to balance resource usage becomes critical.

### How did you solve the problem?
The problem was solved by implementing a dynamic switching mechanism that transitions the object detection inference process from the GPU to the CPU whenever the GPU inference latency exceeds a predefined threshold. The solution involves monitoring the inference time during runtime and comparing it against an inferenceTimeThreshold. If the GPU takes too long, the code creates a new object detection classifier that uses the CPU instead of the GPU.
This new classifier, personClassifierCPU, is initialized with the useGPU parameter set to false and configured to utilize two CPU threads to ensure efficiency. The system then reassigns the imageAnalyzer, which processes camera frames, to use this CPU-based classifier. By dynamically switching the inference process to the CPU, the solution avoids GPU bottlenecks and ensures that object detection continues without significant delays.
The implementation also includes logging to indicate the transition, making it easier to debug the system’s performance. While this approach may slightly increase CPU load, it provides a fallback mechanism to maintain real-time performance when the GPU struggles to handle the computational demand. This dynamic and adaptive strategy addresses the latency issue while optimizing resource utilization across available hardware components.

### How did the inference latency change?
To evaluate the performance increase offered by the optimization, we first executed 10 seconds of the person classifier alone, entered a simple query and waited for LLM generation to finish. We then executed 10 seconds of the person classifier alone again. During the process, we measured the person classifier latency in seconds, every sixth of a second. 
By implementing the optimization as described above, we could observe a significant decrease in latencies during concurrent LLM generation, while not significantly affecting the baseline performance. This indicates that our optimization scheme not only alleviated latencies during parallel execution, but also did not negatively impact application performance when the classifier is run alone. While the optimized version did produce a statistically significant increase in latency during LLM generation, its effect size is dwarfed compared to the unoptimized version.

## Discussion
### Suggest modifications to the TFLite API.
To implement the solution approach into TensorFlow Lite (TFLite), modifications would be required to enable runtime switching between processing units, such as GPU and CPU. Currently, the TFLite API requires developers to specify the processing unit during the initialization of the model, as shown  in  the PersonClassifier.kt  file.  The BaseOptions.builder().setNumThreads()  and
.apply { if (useGPU) useGpu() } configurations define the processing unit before the model is loaded. However, there is no mechanism to dynamically change this configuration during runtime based on performance metrics such as inference latency.
So we suggest a new method, setProcessingUnit, which could be added to the BaseOptions class or as an additional API in TFLite. This method would allow developers to dynamically reconfigure the model’s processing unit during runtime without needing to reinitialize the entire model. The method could look like the following pseudocode on Box 1. (based on Kotlin)

```
function setProcessingUnit(context, useGPU, threadCount): boolean {
    // Ensure existing objectDetector is released before switching
    If objectDetector exists then close it

    // Configure BaseOptions for TFLite
    baseOptions = BaseOptions.builder()
    build baseOptions with arguments threadCount and useGPU

    // Configure ObjectDetectorOptions with the updated BaseOptions
    options = ObjectDetector.ObjectDetectorOptions.builder()
    Set base options of options as baseOptions

    Try {
        // Create and load a new ObjectDetector instance
        Create new objectDetector as an ObjectDetector object created from file and options
            (context, MODEL_NAME, configuredOptions)
        Print  "Switched to " + (useGPU ? "GPU" : "CPU")         // Log success
        return TRUE
    } Catch exception as e {
        Print  "Error switching processing unit: " + e.message   // Log failure
        return FALSE
    }
}
```

### What are the limitations of the current solution approach? Any ideas to address them?
There are two critical limitations on the current solution approach. First, once the PersonClassifier switches from GPU to CPU due to high inference latency, it cannot revert back to GPU mode, even when GPU resources become available (e.g., when the LLM finishes its task). This limitation leads to persistent inefficiencies, as the CPU continues to handle the object detection inference, which is not optimized for such tasks compared to the GPU. Therefore, a method that allows for the switching of a process between GPU and CPU is required for maximum efficiency. To switch an execution context from CPU to GPU, we can allocate a page for loading the program into GPU, and invoke the execution of the program entry point. The same can be said for the conversion to CPU. The below pseudocode provides an elementary implementation of CPU → GPU conversion.
```C
void switch_to_gpu() {
    int gpu_page, cpu_page;
    if ((gpu_page = get_gpu_page(PAL_USER | PAL_ZERO)) < 0) return;

    free_cpu_page(cpu_page);                        /* Free CPU from LLM executable */
    load_gpu_page(gpu_page, llm_model_inference);   /* Load GPU with LLM executable */

    gpu_context_switch();                        /* Create and launch a GPU process */
}
```

Moreover, the current solution uses inference time as the metric to determine whether the PersonClassifier should switch from GPU to CPU. This is a simple but problematic approach because inference time can be influenced by temporary spikes in workload or other transient factors, such as a momentary delay in LLM processing. Making decisions based on such short-term observations could lead to unnecessary switching, adding overhead and instability to the system. To fully account for computational resource availability, various factors, primarily including GPU and CPU usage, as well as global system load must be taken into account. Taking these factors into account, we can also implement an additional feature that blocks the execution of an AI model if neither CPU or GPU is sufficient to run the model (due to high load, overheating, etc.).
```C
enum exec_context { HALT, CPU, GPU };
enum exec_context monitor_system() {
    int gpu_usage = get_gpu_usage();               /* #define GPU_USAGE_THRES  60 */
    int gpu_memory = get_gpu_memory_usage();       /* #define GPU_MEMUSG_THRES 80 */
    int cpu_usage = get_cpu_usage();               /* #define CPU_USAGE_THRES  75 */
    int cpu_core_usage = get_cpu_core_usage();     /* #define CPU_CORUSG_THRES 70 */
    int system_load = get_system_load();           /* #define SYS_LOAD_THRES   85 */

    if (gpu_usage < GPU_USAGE_THRES && gpu_memory < GPU_MEMUSG_THRES) return GPU;
    elif (cpu_usage < CPU_USAGE_THRES && cpu_core_usage < CPU_CORUSG_THRES) return CPU;
    else return HALT;
}
void context_manager() {
    switch (monitor_system()) {
        case GPU: switch_to_gpu();
        case CPU: switch_to_cpu();
        case HALT: thread_block();
}}
```

### What other approaches could be taken to address the delay in computation speed?
We could utilize model pruning, where less critical parts of the neural network are removed, resulting in a smaller and faster model. While pruning may reduce accuracy slightly, it often provides significant speed improvements, making it a suitable trade-off for real-time applications. Additionally, model partitioning offers an innovative way to distribute computation across hardware. By dividing the model into smaller submodels that execute independently, developers could allocate specific components to the GPU or CPU, optimizing resource usage and inference times.

## Conclusion
The implementation of dynamic switching between GPU and CPU is a practical solution to the challenges of running multiple AI models on mobile devices. However, it is essential to consider the limitations and potential trade-offs associated with this approach. By incorporating adaptive thresholds, hybrid inference strategies, and alternative optimization techniques such as quantization and pruning, the solution can be further refined to deliver robust and efficient performance. Extending these ideas to platforms like TensorFlow Lite would make them widely applicable, enabling developers to build smarter, more resource-efficient AI applications. This project underscores the importance of balancing performance and resource utilization in modern AI systems, paving the way for more sophisticated approaches to real-time model inference.
