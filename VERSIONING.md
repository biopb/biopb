# Versioning

BioPB uses **git tags** as the single source of truth for version numbers.

## How It Works

- **Python**: Uses `setuptools_scm` to derive version from git tags
- **Java**: CI injects version from tag via Maven's `-Drevision` flag

## Release Process

1. Ensure all changes are committed and pushed
2. Create and push a version tag:
   ```bash
   git tag v0.2.0
   git push --tags
   ```
3. CI automatically:
   - Builds and publishes Python package to PyPI
   - Builds and publishes Java package to Maven Central

## Local Development

- Python builds will show a dev version (e.g., `0.2.0.dev12+g...`)
- Java builds show `0.0.0-SNAPSHOT` by default

To test a specific version locally:
```bash
# Python
SETUPTOOLS_SCM_PRETEND_VERSION=0.2.0 python -m build .

# Java
mvn -Drevision=0.2.0 package
```