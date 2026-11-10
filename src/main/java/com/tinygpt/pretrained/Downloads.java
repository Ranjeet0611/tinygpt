package com.tinygpt.pretrained;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Downloads a file once, and reuses it after that.
 *
 * <pre>
 *   file already on disk?  ── yes ──► use it
 *            │ no
 *            ▼
 *   download to "file.part" (showing progress) ──► rename to "file"
 * </pre>
 *
 * Downloading to ".part" first means a half-finished download is never mistaken for a complete one.
 *
 * Picture guide: docs/classes/9-chat.md#downloads
 */
public final class Downloads {

    /** The book's 1,100 instruction examples (chapter 7). 200 KB. */
    public static final String INSTRUCTION_DATA_URL =
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json";

    private Downloads() {}

    public static Path ifMissing(String url, Path file) throws IOException, InterruptedException {
        if (Files.exists(file)) {
            return file;
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path partial = file.resolveSibling(file.getFileName() + ".part");
        System.out.println("Downloading " + url);

        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpResponse<InputStream> response = client.send(HttpRequest.newBuilder(URI.create(url)).build(),
                HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            throw new IOException("Download failed with HTTP " + response.statusCode() + ": " + url);
        }
        long total = response.headers().firstValueAsLong("content-length").orElse(-1);

        try (InputStream in = response.body(); OutputStream out = Files.newOutputStream(partial)) {
            byte[] buffer = new byte[1 << 16];
            long done = 0;
            int lastPercent = -1;
            for (int read; (read = in.read(buffer)) != -1; ) {
                out.write(buffer, 0, read);
                done += read;
                int percent = total > 0 ? (int) (100 * done / total) : -1;
                if (percent != lastPercent && percent % 10 == 0) {
                    System.out.printf("  %3d%%  (%,d MB)%n", percent, done >> 20);
                    lastPercent = percent;
                }
            }
        }
        Files.move(partial, file, StandardCopyOption.REPLACE_EXISTING);
        return file;
    }
}
