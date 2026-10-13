package biopb.tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Map;
import java.util.TreeMap;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * The pin store the Python SDK keeps, as a {@link TlsPinStore}:
 * {@code <state dir>/tls-known-hosts.json}, a JSON object mapping
 * {@code host:port} to the pinned certificate as a PEM string.
 *
 * <p>A missing, unreadable or malformed file is an empty store, and an entry
 * that is not a string is ignored. A write reads the file again, adds the entry,
 * and replaces the file atomically (owner-only on POSIX), so a reader never sees
 * half a file. Two processes pinning at the same moment can lose one of the pins
 * (the later replace wins); that one is then pinned again on its next first use.
 */
public final class FileTlsPinStore implements TlsPinStore {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();

    /** Shared by every store in the process: {@link Connection} makes one per dial. */
    private static final Object WRITE_LOCK = new Object();

    private final Path file;

    /** A store over {@code file}; it is created on the first {@link #put}. */
    public FileTlsPinStore(Path file) {
        this.file = file;
    }

    @Override
    public String get(String hostPort) {
        return load().get(hostPort);
    }

    @Override
    public void put(String hostPort, String pem) {
        synchronized (WRITE_LOCK) {
            write(hostPort, pem);
        }
    }

    private void write(String hostPort, String pem) {
        Map<String, String> pins = load();
        pins.put(hostPort, pem);
        JsonObject out = new JsonObject();
        pins.forEach(out::addProperty);
        Path dir = file.toAbsolutePath().getParent();
        Path tmp = null;
        try {
            Files.createDirectories(dir);
            tmp = Files.createTempFile(dir, ".tls-", ".tmp");
            Files.write(tmp, GSON.toJson(out).getBytes(StandardCharsets.UTF_8));
            try {
                Files.setPosixFilePermissions(tmp, PosixFilePermissions.fromString("rw-------"));
            } catch (UnsupportedOperationException notPosix) {
                // The pins are public certificates; owner-only is where the platform has it.
            }
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            if (tmp != null) {
                try {
                    Files.deleteIfExists(tmp);
                } catch (IOException ignored) {
                    // The pin failed already; a leftover temp file is harmless.
                }
            }
            throw new UncheckedIOException("could not write the TLS pin store " + file, e);
        }
    }

    private Map<String, String> load() {
        Map<String, String> pins = new TreeMap<>();
        try {
            String text = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
            JsonElement root = JsonParser.parseString(text);
            if (root.isJsonObject()) {
                for (Map.Entry<String, JsonElement> e : root.getAsJsonObject().entrySet()) {
                    JsonElement v = e.getValue();
                    if (v.isJsonPrimitive() && v.getAsJsonPrimitive().isString()) {
                        pins.put(e.getKey(), v.getAsString());
                    }
                }
            }
        } catch (IOException | RuntimeException absentOrMalformed) {
            return new TreeMap<>();
        }
        return pins;
    }
}
