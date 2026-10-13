package biopb.tensor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * The file pin store against the contract in {@code docs/discovery-contract.md}.
 * {@code known-hosts.json} is the file Python writes
 * ({@code src/test/python/tls_tofu_test.py} reads the same one).
 */
public class FileTlsPinStoreTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private Path file;

    @Before
    public void setUp() {
        file = tmp.getRoot().toPath().resolve("state").resolve("tls-known-hosts.json");
    }

    private String resource(String name) throws IOException {
        try (InputStream in = FileTlsPinStoreTest.class.getResourceAsStream("/tls/" + name)) {
            // A Windows checkout may have turned the PEM files' newlines into CRLF.
            return new String(in.readAllBytes(), StandardCharsets.UTF_8).replace("\r\n", "\n");
        }
    }

    @Test
    public void readsTheFilePythonWrites() throws IOException {
        Files.createDirectories(file.getParent());
        Files.write(file, resource("known-hosts.json").getBytes(StandardCharsets.UTF_8));
        FileTlsPinStore store = new FileTlsPinStore(file);

        Assert.assertEquals(resource("server.pem"), store.get("pinned.example:8815"));
        Assert.assertEquals(resource("othername.pem"), store.get("other.example:8815"));
        Assert.assertNull(store.get("unknown.example:8815"));
    }

    @Test
    public void aPinWrittenHereIsAJsonObjectOfPemStringsAndKeepsTheOthers() throws IOException {
        Files.createDirectories(file.getParent());
        Files.write(file, resource("known-hosts.json").getBytes(StandardCharsets.UTF_8));
        new FileTlsPinStore(file).put("new.example:1", "pem-new");

        JsonObject written = JsonParser.parseString(new String(Files.readAllBytes(file), StandardCharsets.UTF_8))
                .getAsJsonObject();
        Assert.assertEquals(3, written.size());
        Assert.assertEquals("pem-new", written.get("new.example:1").getAsString());
        Assert.assertEquals(resource("server.pem"), written.get("pinned.example:8815").getAsString());
        Assert.assertEquals("a second store sees it", "pem-new", new FileTlsPinStore(file).get("new.example:1"));
    }

    @Test
    public void aMissingOrCorruptFileIsAnEmptyStore() throws IOException {
        FileTlsPinStore store = new FileTlsPinStore(file);
        Assert.assertNull(store.get("a:1"));

        Files.createDirectories(file.getParent());
        for (String garbage : new String[] {"", "not json", "[1, 2]", "{\"a:1\": 7}", "{\"a:1\": "}) {
            Files.write(file, garbage.getBytes(StandardCharsets.UTF_8));
            Assert.assertNull(garbage, store.get("a:1"));
        }

    }

    @Test
    public void aCorruptFileIsNotReplacedAndThePinIsHeldInMemory() throws IOException {
        Files.createDirectories(file.getParent());
        for (String garbage : new String[] {"", "not json", "[1, 2]", "{\"a:1\": "}) {
            Files.write(file, garbage.getBytes(StandardCharsets.UTF_8));
            FileTlsPinStore store = new FileTlsPinStore(file);
            store.put("a:1", "pem-a");
            Assert.assertEquals(garbage, new String(Files.readAllBytes(file), StandardCharsets.UTF_8));
            Assert.assertEquals("held for this store", "pem-a", store.get("a:1"));
            Assert.assertNull("not for another process", new FileTlsPinStore(file).get("a:1"));
        }
    }

    @Test
    public void anUnwritableDirectoryDoesNotFailThePin() throws IOException {
        // A file where the state directory should be: nothing can be created under it.
        Path blocker = tmp.newFile("state-is-a-file").toPath();
        FileTlsPinStore store = new FileTlsPinStore(blocker.resolve("tls-known-hosts.json"));
        store.put("a:1", "pem-a");
        Assert.assertEquals("pem-a", store.get("a:1"));
    }

    @Test
    public void theFileIsOwnerOnlyAndNoTempFileIsLeft() throws IOException {
        new FileTlsPinStore(file).put("a:1", "pem-a");
        try (var names = Files.list(file.getParent())) {
            Assert.assertEquals(List.of("tls-known-hosts.json"),
                    names.map(p -> p.getFileName().toString()).collect(Collectors.toList()));
        }
        if (Files.getFileStore(file).supportsFileAttributeView("posix")) {
            Set<PosixFilePermission> perms = Files.getPosixFilePermissions(file);
            Assert.assertEquals(Set.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE), perms);
        }
    }

    @Test
    public void concurrentWritersLeaveAWholeFile() throws Exception {
        FileTlsPinStore store = new FileTlsPinStore(file);
        ExecutorService pool = Executors.newFixedThreadPool(8);
        try {
            List<Future<?>> done = new ArrayList<>();
            for (int i = 0; i < 40; i++) {
                String key = "host" + i + ":1";
                done.add(pool.submit(() -> store.put(key, "pem-" + key)));
            }
            for (Future<?> f : done) {
                f.get();
            }
        } finally {
            pool.shutdown();
        }
        // Every write is one complete file; this process serializes its writers, so none is lost.
        for (int i = 0; i < 40; i++) {
            Assert.assertEquals("pem-host" + i + ":1", store.get("host" + i + ":1"));
        }
    }

    @Test
    public void aPinSurvivesARestart() throws Exception {
        new FileTlsPinStore(file).put("pinned.example:8815", resource("othername.pem"));
        // A second store over the same file is what a new JVM sees.
        Assert.assertEquals(resource("othername.pem"), new FileTlsPinStore(file).get("pinned.example:8815"));
    }
}
