/** A string as a SQL literal for the catalog's query surface, which takes no parameters. */
export function sqlLiteral(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}
