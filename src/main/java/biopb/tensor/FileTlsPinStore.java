package biopb.tensor;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

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
 * <p>A missing, unreadable or malformed file reads as an empty store, and an
 * entry that is not a string is ignored. A write reads the file again, adds the
 * entry, and replaces the file atomically (owner-only on POSIX), so a reader
 * never sees half a file. It refuses to replace a file that exists but cannot be
 * read as a JSON object, which would drop every other host's pin. Two processes
 * pinning at the same moment can lose one of the pins (the later replace wins);
 * that one is then pinned again on its next first use.
 *
 * <p>A write that fails (a read-only state directory, a full disk, a file it
 * will not replace) is logged, and the pin is kept in memory by this store, so
 * the connection goes on and the pin holds until the process ends.
 */
public final class FileTlsPinStore implements TlsPinStore {

    private static final Logger LOGGER = Logger.getLogger(FileTlsPinStore.class.getName());

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create();

    /** Shared by every store in the process: {@link Connection} makes one per dial. */
    private static final Object WRITE_LOCK = new Object();

    private final Path file;
    private final Map<String, String> unwritten = new ConcurrentHashMap<>();

    /** A store over {@code file}; it is created on the first {@link #put}. */
    public FileTlsPinStore(Path file) {
        this.file = file;
    }

    @Override
    public String get(String hostPort) {
        String pinned = load().get(hostPort);
        return pinned != null ? pinned : unwritten.get(hostPort);
    }

    @Override
    public void put(String hostPort, String pem) {
        try {
            synchronized (WRITE_LOCK) {
                write(hostPort, pem);
            }
        } catch (IOException | RuntimeException e) {
            LOGGER.warning("could not write the TLS pin store " + file + " (" + e + "); pinning " + hostPort
                    + " for this process only");
            unwritten.put(hostPort, pem);
        }
    }

    private void write(String hostPort, String pem) throws IOException {
        Map<String, String> pins = loadForWrite();
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
            throw e;
        }
    }

    /** The pins in the file; empty when it is missing, unreadable or malformed. */
    private Map<String, String> load() {
        try {
            return read();
        } catch (IOException | RuntimeException absentOrMalformed) {
            return new TreeMap<>();
        }
    }

    /** The pins to add to: none for a missing file, an error for one that exists and is not a JSON object. */
    private Map<String, String> loadForWrite() throws IOException {
        try {
            return read();
        } catch (NoSuchFileException absent) {
            return new TreeMap<>();
        } catch (RuntimeException malformed) {
            throw new IOException("not a JSON object of pins: " + file, malformed);
        }
    }

    private Map<String, String> read() throws IOException {
        Map<String, String> pins = new TreeMap<>();
        String text = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
        JsonElement root = JsonParser.parseString(text);
        if (!root.isJsonObject()) {
            throw new IOException("not a JSON object of pins: " + file);
        }
        for (Map.Entry<String, JsonElement> e : root.getAsJsonObject().entrySet()) {
            JsonElement v = e.getValue();
            if (v.isJsonPrimitive() && v.getAsJsonPrimitive().isString()) {
                pins.put(e.getKey(), v.getAsString());
            }
        }
        return pins;
    }
}
