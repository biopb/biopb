/** Which ids are on screen now that were not last time -- the rows the observe
 * page plays an entrance for.
 *
 * Its own function because the rule is easy to state and easy to get subtly
 * wrong, and the poll it lives in is not reachable from a test without a DOM.
 * A null `before` is the first load, where every row is new and none of them
 * arrived: animating a whole list at once says nothing about any of it. */
export function arrivals(before: Set<string> | null, now: Set<string>): string[] {
  if (!before) return [];
  return [...now].filter((id) => !before.has(id));
}
