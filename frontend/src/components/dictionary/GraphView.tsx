import React from "react";

import { drag, forceCenter, forceLink, forceManyBody, forceSimulation, select, zoom } from "d3";

import type { GraphPayload } from "../../api/dictionary";

interface GraphViewProps {
  graph: GraphPayload;
  onSelectNode?: (nodeId: string) => void;
}

type SimNode = GraphPayload["nodes"][number] & {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
};

type SimLink = {
  source: string | SimNode;
  target: string | SimNode;
  relation_type: string;
  weight: number;
};

function categoryColor(category: string | undefined): string {
  const palette = ["#6366f1", "#10b981", "#f59e0b", "#38bdf8", "#f43f5e", "#22c55e", "#eab308"];
  if (!category) return "#94a3b8";
  let hash = 0;
  for (let idx = 0; idx < category.length; idx += 1) {
    hash = (hash << 5) - hash + category.charCodeAt(idx);
    hash |= 0;
  }
  return palette[Math.abs(hash) % palette.length];
}

function nodeRadius(node: SimNode): number {
  const usage = Math.max(0, Number(node.usage_count ?? 0));
  return Math.max(8, Math.min(22, 8 + Math.sqrt(usage + 1) * 1.8));
}

export class GraphView extends React.PureComponent<GraphViewProps> {
  private svgRef = React.createRef<SVGSVGElement>();

  private containerRef = React.createRef<HTMLDivElement>();

  private cleanup: (() => void) | null = null;

  componentDidMount(): void {
    this.drawGraph();
  }

  componentDidUpdate(prevProps: GraphViewProps): void {
    if (prevProps.graph !== this.props.graph || prevProps.onSelectNode !== this.props.onSelectNode) {
      this.drawGraph();
    }
  }

  componentWillUnmount(): void {
    this.cleanup?.();
    this.cleanup = null;
  }

  private drawGraph(): void {
    this.cleanup?.();
    this.cleanup = null;

    const { graph, onSelectNode } = this.props;
    const svgEl = this.svgRef.current;
    const containerEl = this.containerRef.current;
    if (!svgEl || !containerEl || graph.nodes.length === 0) return;

    const width = Math.max(320, containerEl.clientWidth);
    const height = 460;

    const nodes: SimNode[] = graph.nodes.map((node) => ({ ...node }));
    const links: SimLink[] = graph.edges.map((edge) => ({ ...edge }));

    const svg = select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const root = svg.append("g");

    const link = root
      .append("g")
      .attr("stroke", "#334155")
      .attr("stroke-opacity", 0.8)
      .selectAll<SVGLineElement, SimLink>("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke-width", (d: SimLink) => Math.max(1, d.weight));

    const node = root
      .append("g")
      .attr("stroke", "#0f172a")
      .attr("stroke-width", 1.5)
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("r", (d: SimNode) => nodeRadius(d))
      .attr("fill", (d: SimNode) => categoryColor(d.category))
      .attr("cursor", "pointer")
      .on("click", (_event: MouseEvent, d: SimNode) => {
        onSelectNode?.(d.id);
      });

    const label = root
      .append("g")
      .selectAll<SVGTextElement, SimNode>("text")
      .data(nodes)
      .enter()
      .append("text")
      .text((d: SimNode) => d.label)
      .attr("font-size", 11)
      .attr("fill", "#e2e8f0")
      .attr("text-anchor", "middle")
      .attr("dy", 4)
      .attr("pointer-events", "none");

    const simulation = forceSimulation(nodes)
      .force(
        "link",
        forceLink<SimNode, SimLink>(links)
          .id((d: SimNode) => d.id)
          .distance(95)
          .strength(0.2)
      )
      .force("charge", forceManyBody().strength(-230))
      .force("center", forceCenter(width / 2, height / 2));

    simulation.on("tick", () => {
      link
        .attr("x1", (d: SimLink) => ((d.source as SimNode).x ?? 0))
        .attr("y1", (d: SimLink) => ((d.source as SimNode).y ?? 0))
        .attr("x2", (d: SimLink) => ((d.target as SimNode).x ?? 0))
        .attr("y2", (d: SimLink) => ((d.target as SimNode).y ?? 0));

      node.attr("cx", (d: SimNode) => d.x ?? 0).attr("cy", (d: SimNode) => d.y ?? 0);
      label.attr("x", (d: SimNode) => d.x ?? 0).attr("y", (d: SimNode) => d.y ?? 0);
    });

    node.call(
      drag<SVGCircleElement, SimNode>()
        .on("start", (event: any, d: SimNode) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event: any, d: SimNode) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event: any, d: SimNode) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
    );

    svg.call(
      zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.4, 2.8])
        .on("zoom", (event: any) => {
          root.attr("transform", event.transform);
        })
    );

    this.cleanup = () => {
      simulation.stop();
      svg.selectAll("*").remove();
    };
  }

  render(): JSX.Element {
    const { graph } = this.props;
    const summary = { nodes: graph.nodes.length, edges: graph.edges.length };

    return (
      <div className="card p-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-heading text-lg">Relation Graph</h3>
          <span className="font-mono text-xs text-slate-400">
            {summary.nodes} nodes • {summary.edges} edges
          </span>
        </div>
        <div
          ref={this.containerRef}
          className="relative min-h-[460px] overflow-hidden rounded-card border border-slate-700/60 bg-slate-900/50"
        >
          {summary.nodes === 0 ? (
            <p className="p-5 text-sm text-slate-400">No graph data yet.</p>
          ) : (
            <svg ref={this.svgRef} className="h-[460px] w-full" aria-label="Dictionary relation graph" />
          )}
        </div>
      </div>
    );
  }
}
