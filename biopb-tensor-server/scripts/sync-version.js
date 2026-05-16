const fs = require('fs');
const path = require('path');

const version = fs.readFileSync(path.join(__dirname, '../VERSION'), 'utf8').trim();

const packages = [
  '../packages/web/package.json',
  '../packages/tensor-flight-client/package.json',
];

for (const pkg of packages) {
  const pkgPath = path.join(__dirname, pkg);
  const pkgJson = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  pkgJson.version = version;
  fs.writeFileSync(pkgPath, JSON.stringify(pkgJson, null, 2) + '\n');
  console.log(`Updated ${pkgPath} to version ${version}`);
}