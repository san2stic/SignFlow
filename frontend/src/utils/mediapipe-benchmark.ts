/**
 * MediaPipe Performance Benchmark Suite
 *
 * Usage:
 *   const benchmark = new MediaPipeBenchmark();
 *   await benchmark.runAll();
 *   console.log(benchmark.getReport());
 */

interface BenchmarkResult {
  name: string;
  duration: number; // ms
  fps: number;
  avgLatency: number; // ms
  p95Latency: number; // ms
  p99Latency: number; // ms
  droppedFrames: number;
  memoryUsed: number; // MB
  gcCount: number;
  cacheHitRate?: number; // 0-1
}

interface BenchmarkScenario {
  name: string;
  duration: number; // seconds
  config: {
    targetFps: number;
    modelComplexity: 0 | 1 | 2;
    adaptiveQuality?: boolean;
    adaptiveFps?: boolean;
  };
  videoSource: "static" | "dynamic" | "rapid" | "lowlight";
}

export class MediaPipeBenchmark {
  private results: BenchmarkResult[] = [];
  private gcObserver: PerformanceObserver | null = null;

  constructor() {
    // Setup GC observer if available
    if (typeof PerformanceObserver !== "undefined") {
      try {
        this.gcObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === "measure") {
              // Track GC events
            }
          }
        });
        this.gcObserver.observe({ entryTypes: ["measure"] });
      } catch {
        // GC observation not supported
      }
    }
  }

  /**
   * Predefined scenarios
   */
  private scenarios: BenchmarkScenario[] = [
    {
      name: "Baseline - Static Pose (30fps, quality 2)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: false,
        adaptiveFps: false
      },
      videoSource: "static"
    },
    {
      name: "Optimized - Static Pose (adaptive)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: true,
        adaptiveFps: true
      },
      videoSource: "static"
    },
    {
      name: "Baseline - Dynamic Signs (30fps, quality 2)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: false,
        adaptiveFps: false
      },
      videoSource: "dynamic"
    },
    {
      name: "Optimized - Dynamic Signs (adaptive)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: true,
        adaptiveFps: true
      },
      videoSource: "dynamic"
    },
    {
      name: "Baseline - Rapid Movement (30fps, quality 2)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: false,
        adaptiveFps: false
      },
      videoSource: "rapid"
    },
    {
      name: "Optimized - Rapid Movement (adaptive)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: true,
        adaptiveFps: true
      },
      videoSource: "rapid"
    },
    {
      name: "Baseline - Low Light (30fps, quality 2)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: false,
        adaptiveFps: false
      },
      videoSource: "lowlight"
    },
    {
      name: "Optimized - Low Light (adaptive)",
      duration: 30,
      config: {
        targetFps: 30,
        modelComplexity: 2,
        adaptiveQuality: true,
        adaptiveFps: true
      },
      videoSource: "lowlight"
    }
  ];

  /**
   * Run single benchmark scenario
   */
  async runScenario(scenario: BenchmarkScenario): Promise<BenchmarkResult> {
    console.log(`[Benchmark] Running: ${scenario.name}`);

    const latencies: number[] = [];
    let droppedFrames = 0;
    let frameCount = 0;
    const startMemory = (performance as any).memory?.usedJSHeapSize ?? 0;
    const startGcCount = this.getGcCount();

    const startTime = performance.now();
    const endTime = startTime + scenario.duration * 1000;

    // Simulate MediaPipe processing loop
    while (performance.now() < endTime) {
      const frameStart = performance.now();

      // Simulate processing (replace with actual MediaPipe call in real usage)
      await this.simulateFrame(scenario.config.modelComplexity);

      const frameEnd = performance.now();
      const latency = frameEnd - frameStart;

      latencies.push(latency);
      frameCount++;

      // Check for dropped frames
      const targetFrameTime = 1000 / scenario.config.targetFps;
      if (latency > targetFrameTime) {
        droppedFrames++;
      }

      // Wait for next frame
      const remainingTime = targetFrameTime - latency;
      if (remainingTime > 0) {
        await this.sleep(remainingTime);
      }
    }

    const totalDuration = performance.now() - startTime;
    const endMemory = (performance as any).memory?.usedJSHeapSize ?? 0;
    const endGcCount = this.getGcCount();

    // Calculate statistics
    latencies.sort((a, b) => a - b);
    const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const p95Index = Math.floor(latencies.length * 0.95);
    const p99Index = Math.floor(latencies.length * 0.99);

    const result: BenchmarkResult = {
      name: scenario.name,
      duration: totalDuration,
      fps: (frameCount / (totalDuration / 1000)),
      avgLatency,
      p95Latency: latencies[p95Index] ?? avgLatency,
      p99Latency: latencies[p99Index] ?? avgLatency,
      droppedFrames,
      memoryUsed: (endMemory - startMemory) / 1024 / 1024, // MB
      gcCount: endGcCount - startGcCount
    };

    this.results.push(result);
    console.log(`[Benchmark] Completed: ${scenario.name}`);

    return result;
  }

  /**
   * Run all scenarios
   */
  async runAll(): Promise<void> {
    console.log("[Benchmark] Starting full benchmark suite...");

    for (const scenario of this.scenarios) {
      await this.runScenario(scenario);

      // Cool down between tests
      await this.sleep(2000);
    }

    console.log("[Benchmark] All scenarios completed!");
  }

  /**
   * Get formatted report
   */
  getReport(): string {
    const lines: string[] = [];

    lines.push("# MediaPipe Performance Benchmark Report");
    lines.push("");
    lines.push(`Generated: ${new Date().toISOString()}`);
    lines.push(`Browser: ${navigator.userAgent}`);
    lines.push("");

    // Group by scenario type
    const grouped = this.groupResults();

    for (const [group, results] of Object.entries(grouped)) {
      lines.push(`## ${group}`);
      lines.push("");

      // Create comparison table
      lines.push("| Metric | Baseline | Optimized | Improvement |");
      lines.push("|--------|----------|-----------|-------------|");

      const baseline = results.find(r => r.name.includes("Baseline"));
      const optimized = results.find(r => r.name.includes("Optimized"));

      if (baseline && optimized) {
        const metrics = [
          {
            name: "FPS",
            baseline: baseline.fps.toFixed(1),
            optimized: optimized.fps.toFixed(1),
            improvement: this.calcImprovement(baseline.fps, optimized.fps, true)
          },
          {
            name: "Avg Latency (ms)",
            baseline: baseline.avgLatency.toFixed(1),
            optimized: optimized.avgLatency.toFixed(1),
            improvement: this.calcImprovement(baseline.avgLatency, optimized.avgLatency, false)
          },
          {
            name: "P95 Latency (ms)",
            baseline: baseline.p95Latency.toFixed(1),
            optimized: optimized.p95Latency.toFixed(1),
            improvement: this.calcImprovement(baseline.p95Latency, optimized.p95Latency, false)
          },
          {
            name: "P99 Latency (ms)",
            baseline: baseline.p99Latency.toFixed(1),
            optimized: optimized.p99Latency.toFixed(1),
            improvement: this.calcImprovement(baseline.p99Latency, optimized.p99Latency, false)
          },
          {
            name: "Dropped Frames",
            baseline: baseline.droppedFrames.toString(),
            optimized: optimized.droppedFrames.toString(),
            improvement: this.calcImprovement(baseline.droppedFrames, optimized.droppedFrames, false)
          },
          {
            name: "Memory (MB)",
            baseline: baseline.memoryUsed.toFixed(1),
            optimized: optimized.memoryUsed.toFixed(1),
            improvement: this.calcImprovement(baseline.memoryUsed, optimized.memoryUsed, false)
          },
          {
            name: "GC Count",
            baseline: baseline.gcCount.toString(),
            optimized: optimized.gcCount.toString(),
            improvement: this.calcImprovement(baseline.gcCount, optimized.gcCount, false)
          }
        ];

        for (const metric of metrics) {
          lines.push(
            `| ${metric.name} | ${metric.baseline} | ${metric.optimized} | ${metric.improvement} |`
          );
        }
      }

      lines.push("");
    }

    // Overall summary
    lines.push("## Overall Summary");
    lines.push("");
    lines.push("Average improvements across all scenarios:");
    lines.push("");

    const avgImprovements = this.calculateAverageImprovements();
    for (const [metric, improvement] of Object.entries(avgImprovements)) {
      lines.push(`- **${metric}**: ${improvement}`);
    }

    return lines.join("\n");
  }

  /**
   * Export results as JSON
   */
  exportJSON(): string {
    return JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        browser: navigator.userAgent,
        results: this.results
      },
      null,
      2
    );
  }

  /**
   * Clear results
   */
  clear(): void {
    this.results = [];
  }

  // Helper methods

  private async simulateFrame(complexity: 0 | 1 | 2): Promise<void> {
    // Simulate processing time based on complexity
    const baseTime = 20; // ms
    const complexityMultiplier = [0.5, 1, 1.5][complexity];
    const processingTime = baseTime * complexityMultiplier;

    await this.sleep(processingTime);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private getGcCount(): number {
    // Try to estimate GC count (browser-specific)
    if ((performance as any).memory) {
      // Rough heuristic: significant drops in usedJSHeapSize
      return 0; // Placeholder
    }
    return 0;
  }

  private groupResults(): Record<string, BenchmarkResult[]> {
    const grouped: Record<string, BenchmarkResult[]> = {};

    for (const result of this.results) {
      const group = result.name.split(" - ")[1]?.split(" (")[0] ?? "Unknown";

      if (!grouped[group]) {
        grouped[group] = [];
      }

      grouped[group].push(result);
    }

    return grouped;
  }

  private calcImprovement(
    baseline: number,
    optimized: number,
    higherIsBetter: boolean
  ): string {
    const change = higherIsBetter
      ? ((optimized - baseline) / baseline) * 100
      : ((baseline - optimized) / baseline) * 100;

    const sign = change > 0 ? "+" : "";
    const emoji = change > 0 ? "🟢" : change < 0 ? "🔴" : "⚪";

    return `${emoji} ${sign}${change.toFixed(1)}%`;
  }

  private calculateAverageImprovements(): Record<string, string> {
    const baselineResults = this.results.filter(r => r.name.includes("Baseline"));
    const optimizedResults = this.results.filter(r => r.name.includes("Optimized"));

    if (baselineResults.length === 0 || optimizedResults.length === 0) {
      return {};
    }

    const metrics = ["fps", "avgLatency", "droppedFrames", "memoryUsed", "gcCount"] as const;
    const improvements: Record<string, string> = {};

    for (const metric of metrics) {
      const baselineAvg =
        baselineResults.reduce((sum, r) => sum + (r[metric] as number), 0) / baselineResults.length;
      const optimizedAvg =
        optimizedResults.reduce((sum, r) => sum + (r[metric] as number), 0) / optimizedResults.length;

      const higherIsBetter = metric === "fps";
      improvements[metric] = this.calcImprovement(baselineAvg, optimizedAvg, higherIsBetter);
    }

    return improvements;
  }
}

// Singleton instance
export const benchmark = new MediaPipeBenchmark();

// Auto-run on dev mode
if (import.meta.env.DEV && import.meta.env.VITE_RUN_BENCHMARK === "true") {
  console.log("[Benchmark] Auto-running benchmark suite...");
  benchmark.runAll().then(() => {
    console.log(benchmark.getReport());
  });
}
