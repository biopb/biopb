const cp = require('child_process');
const fs = require('fs');
const path = require('path');

// Stamp the web packages with the PRODUCT release version. The web bundle ships
// as part of the product (control + mcp + the tensor-server wheel and image), so
// it shares the product `release-v*` git tag — NOT the SDK's `v*`. (The tensor
// server now tracks `release-v*` too, so there is no longer a separate
// `server-v*` line.) `git describe --abbrev=0 --match "release-v*"` returns the
// nearest release tag *name* (no `-<N>-g<sha>` suffix), so at a clean
// `release-vX.Y.Z` tag this is exactly `X.Y.Z` and between releases it is the last
// release version. These packages are `private` (never published), so the base
// version is all that's meaningful — no dev/commit suffix is appended.
function releaseVersion() {
  try {
    const tag = cp
      .execSync('git describe --tags --abbrev=0 --match "release-v*"', {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
      })
      .trim();
    const m = tag.match(/^release-v(.+)$/);
    if (m) return m[1];
  } catch {
    // No release-v* tag reachable (fresh clone / shallow checkout without tags).
  }
  return '0.0.0';
}

const version = releaseVersion();

const packages = [
  '../packages/app/package.json',
  '../packages/tensor-flight-client/package.json',
];

for (const pkg of packages) {
  const pkgPath = path.join(__dirname, pkg);
  const pkgJson = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  pkgJson.version = version;
  fs.writeFileSync(pkgPath, JSON.stringify(pkgJson, null, 2) + '\n');
  console.log(`Updated ${pkgPath} to version ${version}`);
}
