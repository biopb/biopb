const fs = require('fs');
const path = require('path');

// Read version from setuptools_scm-generated _version.py
const versionPyPath = path.join(__dirname, '../biopb_tensor_server/_version.py');

let version;
try {
  const versionPy = fs.readFileSync(versionPyPath, 'utf8');
  const match = versionPy.match(/__version__ = "([^"]+)"/);
  version = match ? match[1] : '0.0.0.dev0';
} catch {
  // _version.py not found (likely not built yet) - fallback to git
  const { execSync } = require('child_process');
  try {
    // Get version from git tags matching server-v*
    const tag = execSync('git describe --tags --match "server-v*" 2>/dev/null || echo "server-v0.0.0"', { encoding: 'utf8' }).trim();
    const match = tag.match(/^server-v(\d+\.\d+\.\d+)/);
    version = match ? match[1] : '0.0.0.dev0';
    // Handle dev versions (commits after tag)
    if (tag.includes('-')) {
      const parts = tag.split('-');
      const devN = parts[parts.length - 1].split('.')[0];
      version = `${version}.dev${devN}`;
    }
  } catch {
    version = '0.0.0.dev0';
  }
}

const packages = [
  '../packages/web/package.json',
  '../packages/tensor-flight-client/package.json',
];

for (const pkg of packages) {
  const pkgPath = path.join(__dirname, pkg);
  const pkgJson = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  pkgJson.version = version.replace(/\.dev\d+$/, ''); // strip .dev suffix for npm
  fs.writeFileSync(pkgPath, JSON.stringify(pkgJson, null, 2) + '\n');
  console.log(`Updated ${pkgPath} to version ${pkgJson.version}`);
}