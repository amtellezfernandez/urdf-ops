/**
 * MetricsChart - lightweight SVG time series visualization.
 */

import { useMemo } from "react";
import { cn } from "@/shared/lib/utils";

import { useMetricsStore, selectVisibleSeries } from "./useMetricsStore";
import type { MetricSeries, ChartConfig } from "./types";
import {
  DEFAULT_CHART_HEIGHT_PX,
  EMPTY_STATE_HEIGHT_PX,
  GRID_LINE_COUNT,
  MIN_VALUE_RANGE,
  SVG_PADDING_BOTTOM,
  SVG_PADDING_LEFT,
  SVG_PADDING_RIGHT,
  SVG_PADDING_TOP,
  SVG_VIEWBOX_HEIGHT,
  SVG_VIEWBOX_WIDTH,
} from "./metricsChartParams";

interface MetricsChartProps {
  className?: string;
  height?: number;
  series?: MetricSeries[];
  config?: Partial<ChartConfig>;
}

type ChartPoint = {
  x: number;
  y: number;
};

function applySmoothing(data: number[], factor: number): number[] {
  if (factor === 0 || data.length === 0) return data;

  const smoothed: number[] = [data[0]];
  for (let i = 1; i < data.length; i++) {
    smoothed.push(factor * data[i] + (1 - factor) * smoothed[i - 1]);
  }
  return smoothed;
}

function formatAxisValue(value: number): string {
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
  return value.toFixed(2);
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatXAxisValue(value: number, xAxis: ChartConfig["xAxis"]): string {
  if (xAxis === "time") return formatTimestamp(value);
  if (xAxis === "epoch") return `E${value}`;
  return `${value}`;
}

function toPath(points: ChartPoint[]): string {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
}

export function MetricsChart({
  className,
  height = DEFAULT_CHART_HEIGHT_PX,
  series: propSeries,
  config: propConfig,
}: MetricsChartProps) {
  const storeSeries = useMetricsStore(selectVisibleSeries);
  const { chartConfig: storeConfig } = useMetricsStore();

  const series = propSeries || storeSeries;
  const config = { ...storeConfig, ...propConfig };
  const xKey = config.xAxis === "time" ? "timestamp" : config.xAxis === "epoch" ? "epoch" : "step";

  const chartSeries = useMemo(() => {
    return series
      .filter((candidate) => candidate.data.length > 0)
      .map((candidate) => {
        const sortedData = [...candidate.data].sort((a, b) => a[xKey] - b[xKey]);
        const smoothedValues = applySmoothing(
          sortedData.map((point) => point.value),
          config.smoothing
        );
        return {
          ...candidate,
          data: sortedData.map((point, index) => ({
            ...point,
            value: smoothedValues[index],
          })),
        };
      });
  }, [series, xKey, config.smoothing]);

  const chartBounds = useMemo(() => {
    const allPoints = chartSeries.flatMap((candidate) => candidate.data);
    if (allPoints.length === 0) return null;

    const xValues = allPoints.map((point) => point[xKey]);
    const yValues = allPoints.map((point) => point.value);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    const minY = Math.min(...yValues);
    const maxY = Math.max(...yValues);
    const yRange = Math.max(maxY - minY, MIN_VALUE_RANGE);
    const yPadding = yRange * 0.08;

    return {
      minX,
      maxX: Math.max(maxX, minX + MIN_VALUE_RANGE),
      minY: minY - yPadding,
      maxY: maxY + yPadding,
    };
  }, [chartSeries, xKey]);

  const plottedSeries = useMemo(() => {
    if (!chartBounds) return [];

    const plotWidth = SVG_VIEWBOX_WIDTH - SVG_PADDING_LEFT - SVG_PADDING_RIGHT;
    const plotHeight = SVG_VIEWBOX_HEIGHT - SVG_PADDING_TOP - SVG_PADDING_BOTTOM;
    const xRange = chartBounds.maxX - chartBounds.minX;
    const yRange = Math.max(chartBounds.maxY - chartBounds.minY, MIN_VALUE_RANGE);

    return chartSeries.map((candidate) => ({
      ...candidate,
      points: candidate.data.map((point) => ({
        x: SVG_PADDING_LEFT + ((point[xKey] - chartBounds.minX) / xRange) * plotWidth,
        y: SVG_PADDING_TOP + (1 - (point.value - chartBounds.minY) / yRange) * plotHeight,
      })),
    }));
  }, [chartBounds, chartSeries, xKey]);

  const gridTicks = useMemo(() => {
    if (!chartBounds) return [];

    const plotHeight = SVG_VIEWBOX_HEIGHT - SVG_PADDING_TOP - SVG_PADDING_BOTTOM;
    const valueRange = chartBounds.maxY - chartBounds.minY;
    return Array.from({ length: GRID_LINE_COUNT + 1 }, (_, index) => {
      const ratio = index / GRID_LINE_COUNT;
      return {
        y: SVG_PADDING_TOP + ratio * plotHeight,
        value: chartBounds.maxY - ratio * valueRange,
      };
    });
  }, [chartBounds]);

  if (series.length === 0 || !chartBounds) {
    return (
      <div
        className={cn(
          "flex items-center justify-center bg-muted/30 rounded-lg text-muted-foreground",
          className
        )}
        style={{ height: height || EMPTY_STATE_HEIGHT_PX }}
      >
        <p className="text-sm">No metrics data to display</p>
      </div>
    );
  }

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <svg
        role="img"
        aria-label="Training metrics chart"
        className="h-full w-full overflow-visible"
        viewBox={`0 0 ${SVG_VIEWBOX_WIDTH} ${SVG_VIEWBOX_HEIGHT}`}
        preserveAspectRatio="none"
      >
        {config.showGrid
          ? gridTicks.map((tick) => (
              <g key={tick.y}>
                <line
                  x1={SVG_PADDING_LEFT}
                  x2={SVG_VIEWBOX_WIDTH - SVG_PADDING_RIGHT}
                  y1={tick.y}
                  y2={tick.y}
                  stroke="currentColor"
                  strokeOpacity={0.12}
                />
                <text
                  x={SVG_PADDING_LEFT - 8}
                  y={tick.y + 4}
                  textAnchor="end"
                  className="fill-muted-foreground text-[10px]"
                  vectorEffect="non-scaling-stroke"
                >
                  {formatAxisValue(tick.value)}
                </text>
              </g>
            ))
          : null}

        <line
          x1={SVG_PADDING_LEFT}
          x2={SVG_VIEWBOX_WIDTH - SVG_PADDING_RIGHT}
          y1={SVG_VIEWBOX_HEIGHT - SVG_PADDING_BOTTOM}
          y2={SVG_VIEWBOX_HEIGHT - SVG_PADDING_BOTTOM}
          stroke="currentColor"
          strokeOpacity={0.24}
        />
        <line
          x1={SVG_PADDING_LEFT}
          x2={SVG_PADDING_LEFT}
          y1={SVG_PADDING_TOP}
          y2={SVG_VIEWBOX_HEIGHT - SVG_PADDING_BOTTOM}
          stroke="currentColor"
          strokeOpacity={0.24}
        />

        <text
          x={SVG_PADDING_LEFT}
          y={SVG_VIEWBOX_HEIGHT - 10}
          textAnchor="start"
          className="fill-muted-foreground text-[10px]"
        >
          {formatXAxisValue(chartBounds.minX, config.xAxis)}
        </text>
        <text
          x={SVG_VIEWBOX_WIDTH - SVG_PADDING_RIGHT}
          y={SVG_VIEWBOX_HEIGHT - 10}
          textAnchor="end"
          className="fill-muted-foreground text-[10px]"
        >
          {formatXAxisValue(chartBounds.maxX, config.xAxis)}
        </text>

        {plottedSeries.map((candidate) => (
          <path
            key={candidate.name}
            d={toPath(candidate.points)}
            fill="none"
            stroke={candidate.color}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>

      {config.showLegend ? (
        <div className="-mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 px-2 text-xs text-muted-foreground">
          {plottedSeries.map((candidate) => (
            <div key={candidate.name} className="flex items-center gap-2">
              <span
                aria-hidden="true"
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: candidate.color }}
              />
              <span>{candidate.label}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
