export type RunnerMode = "synthetic" | "adapter";

export type ModelResolution = {
  requestedModel: string;
  effectiveModel: string;
  remapped: boolean;
};

export function resolveRuntimeModel(input: {
  requestedModel?: string;
  backendModel: string;
  canonicalModel: string;
  mode: RunnerMode;
}): ModelResolution {
  const requestedModel = input.requestedModel?.trim() || input.canonicalModel;
  if (input.mode === "adapter") {
    const effectiveModel = input.backendModel.trim() || requestedModel;
    return {
      requestedModel,
      effectiveModel,
      remapped: effectiveModel !== requestedModel,
    };
  }

  return {
    requestedModel,
    effectiveModel: requestedModel,
    remapped: false,
  };
}
