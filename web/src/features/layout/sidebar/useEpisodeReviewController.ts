import { useCallback, useEffect } from "react";
import { toast } from "sonner";

import { toAnimationFrames, type Episode, type RecordedFrame } from "@/features/dataset";
import { EPISODE_VELOCITY_LIMIT_TOLERANCE } from "@/features/dataset/episodeReviewParams";
import {
  applyTargetFpsToEpisodes,
  applyEpisodeLimitCorrections,
  buildEpisodeDataExport,
  buildEpisodeSaveResult,
  computeEpisodeFps,
  type SaveEpisodeResult,
} from "@/features/layout/sidebar/episodeReviewHelpers";
import { computeVelocityStatusForFrames } from "@/features/layout/sidebar/sidebarHelpers";
import { viewerPlayback } from "@/features/viewer/playback/viewerPlayback";
import type { LoadedReplayEpisode } from "@/features/layout/sidebar/useReplaySessionController";
import type { JointLimits } from "@/shared/lib/urdfBrowser";
import type { JointLimitMode } from "@/shared/types/feature";
import type { EpisodeSaveHandler } from "@/shared/types/feature";
import type { Dispatch, MutableRefObject, SetStateAction } from "react";

type UseEpisodeReviewControllerParams = {
  jointLimits: JointLimits;
  robotBaseName: string;
  targetFps: number;
  currentFrame?: number;
  setEpisodes: Dispatch<SetStateAction<Episode[]>>;
  setCurrentPlayingEpisodeIndex: Dispatch<SetStateAction<number | null>>;
  currentLoadedEpisodeRef: MutableRefObject<LoadedReplayEpisode | null>;
  isPlayingAllRef: MutableRefObject<boolean>;
  getJointOrderForFrames: (frames: RecordedFrame[]) => string[];
  onEpisodeSaveHandlerChange?: (handler: EpisodeSaveHandler | undefined) => void;
  onViewerEpisodeChange?: (episode: Episode | null) => void;
  onViewerOpenChange?: (open: boolean) => void;
  onViewerSplitViewChange?: (splitView: boolean) => void;
};

export const useEpisodeReviewController = ({
  jointLimits,
  robotBaseName,
  targetFps,
  currentFrame,
  setEpisodes,
  setCurrentPlayingEpisodeIndex,
  currentLoadedEpisodeRef,
  isPlayingAllRef,
  getJointOrderForFrames,
  onEpisodeSaveHandlerChange,
  onViewerEpisodeChange,
  onViewerOpenChange,
  onViewerSplitViewChange,
}: UseEpisodeReviewControllerParams) => {
  const getEpisodeVelocityStatus = useCallback(
    (episode: Episode) =>
      computeVelocityStatusForFrames(
        episode.frames,
        jointLimits,
        EPISODE_VELOCITY_LIMIT_TOLERANCE
      ),
    [jointLimits]
  );

  const handleEpisodeSave = useCallback<EpisodeSaveHandler>(
    (episodeToSave: Episode, saveAsNew: boolean, newName?: string) => {
      if (!episodeToSave || episodeToSave.frames.length === 0) {
        toast.error("Episode has no frames to save");
        return;
      }

      const now = Date.now();
      const trimmedName = newName?.trim();
      let saveResult: SaveEpisodeResult | null = null;

      setEpisodes((previousEpisodes) => {
        saveResult = buildEpisodeSaveResult({
          previousEpisodes,
          episodeToSave,
          saveAsNew,
          newName,
          now,
        });
        return saveResult.episodes;
      });

      if (!saveResult) {
        return;
      }

      if (saveResult.errorMessage) {
        toast.error(saveResult.errorMessage);
        return;
      }

      const savedEpisode = saveResult.savedEpisode;
      if (!savedEpisode) {
        return;
      }

      onViewerSplitViewChange?.(true);
      onViewerOpenChange?.(true);
      onViewerEpisodeChange?.(savedEpisode);

      const resumeFrame = Math.max(
        0,
        Math.min(currentFrame ?? 0, savedEpisode.frames.length - 1)
      );
      viewerPlayback.playEpisode(toAnimationFrames(savedEpisode), {
        autoplay: isPlayingAllRef.current,
        startFrame: resumeFrame,
        playbackEpisode: savedEpisode,
      });
      currentLoadedEpisodeRef.current = {
        index: savedEpisode.number - 1,
        episodeId: savedEpisode.id,
        framesRef: savedEpisode.frames,
      };

      if (saveAsNew) {
        setCurrentPlayingEpisodeIndex(savedEpisode.number - 1);
      }

      toast.success(
        saveAsNew
          ? `Saved ${trimmedName || `Episode ${savedEpisode.number}`}`
          : `Episode ${savedEpisode.number} updated`
      );
    },
    [
      currentFrame,
      currentLoadedEpisodeRef,
      isPlayingAllRef,
      onViewerEpisodeChange,
      onViewerOpenChange,
      onViewerSplitViewChange,
      setCurrentPlayingEpisodeIndex,
      setEpisodes,
    ]
  );

  useEffect(() => {
    if (!onEpisodeSaveHandlerChange) return;
    onEpisodeSaveHandlerChange(handleEpisodeSave);
    return () => onEpisodeSaveHandlerChange(undefined);
  }, [handleEpisodeSave, onEpisodeSaveHandlerChange]);

  const applyTargetFps = useCallback(() => {
    if (!Number.isFinite(targetFps) || targetFps <= 0) {
      toast.error("Enter a valid target FPS");
      return;
    }

    let updatedCount = 0;
    setEpisodes((previousEpisodes) => {
      const result = applyTargetFpsToEpisodes({
        episodes: previousEpisodes,
        targetFps,
      });
      updatedCount = result.updatedCount;
      return result.episodes;
    });

    if (updatedCount === 0) {
      toast.info("All episodes already match target FPS");
      return;
    }

    toast.success(`Applied ${targetFps} FPS to ${updatedCount} episode(s)`);
  }, [setEpisodes, targetFps]);

  const applyLimitCorrections = useCallback(
    (
      frames: RecordedFrame[],
      modeByJoint: Record<string, JointLimitMode | undefined> = {},
      limitsOverride?: JointLimits
    ) =>
      applyEpisodeLimitCorrections(
        frames,
        jointLimits,
        modeByJoint,
        limitsOverride
      ),
    [jointLimits]
  );

  const exportEpisodeToDataFile = useCallback(
    (episode: Episode) => {
      if (episode.frames.length === 0) {
        toast.error("No recorded data to export");
        return;
      }

      const { filename, content } = buildEpisodeDataExport({
        episode,
        robotBaseName,
        getJointOrderForFrames,
      });
      const blob = new Blob([content], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported Episode ${episode.number} to ${filename}`);
    },
    [getJointOrderForFrames, robotBaseName]
  );

  return {
    computeEpisodeFps,
    getEpisodeVelocityStatus,
    applyTargetFps,
    applyLimitCorrections,
    exportEpisodeToDataFile,
  };
};
