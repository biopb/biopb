package biopb.tensor;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.function.Function;

/**
 * The process environment and home directory discovery reads, behind a seam.
 *
 * <p>Java cannot set an environment variable, so a test could never drive the
 * variables the discovery contract names ({@code docs/discovery-contract.md})
 * if they were read straight from {@link System#getenv}.
 */
final class DiscoveryEnvironment {

    private final Function<String, String> variables;
    private final Path home;

    private DiscoveryEnvironment(Function<String, String> variables, Path home) {
        this.variables = variables;
        this.home = home;
    }

    static DiscoveryEnvironment system() {
        return new DiscoveryEnvironment(System::getenv, Paths.get(System.getProperty("user.home")));
    }

    static DiscoveryEnvironment of(Map<String, String> variables, Path home) {
        return new DiscoveryEnvironment(variables::get, home);
    }

    /** The variable's value, or null when unset. */
    String get(String name) {
        return variables.apply(name);
    }

    /** The variable's value with surrounding whitespace removed; empty when unset. */
    String trimmed(String name) {
        String value = get(name);
        return value == null ? "" : value.trim();
    }

    Path home() {
        return home;
    }

    /**
     * The state directory: {@code $BIOPB_STATE_HOME/biopb} when set (it must be
     * absolute), else {@code ~/.local/state/biopb}, on every platform. {@code XDG_*}
     * is not read.
     */
    Path stateDir() {
        String raw = get("BIOPB_STATE_HOME");
        if (raw != null && !raw.isEmpty()) {
            Path base = Paths.get(raw);
            if (!base.isAbsolute()) {
                // A relative value would resolve against each process's own
                // working directory, so processes that must agree would not.
                throw new IllegalArgumentException("BIOPB_STATE_HOME must be an absolute path (got '" + raw + "')");
            }
            return base.resolve("biopb");
        }
        return home.resolve(".local").resolve("state").resolve("biopb");
    }
}
