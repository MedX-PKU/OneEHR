import type { AnalysisModuleMeta } from './types'

const MODULE_PRIORITY = [
  'test_audit',
  'prediction_audit',
  'cohort_analysis',
  'dataset_profile',
  'temporal_analysis',
  'interpretability',
]

function moduleRank(name: string): number {
  const index = MODULE_PRIORITY.indexOf(name)
  return index === -1 ? MODULE_PRIORITY.length : index
}

export function sortModulesByPriority<T extends Pick<AnalysisModuleMeta, 'name'>>(modules: T[]): T[] {
  return [...modules].sort((left, right) => {
    const rankDelta = moduleRank(left.name) - moduleRank(right.name)
    if (rankDelta !== 0) {
      return rankDelta
    }
    return left.name.localeCompare(right.name)
  })
}
