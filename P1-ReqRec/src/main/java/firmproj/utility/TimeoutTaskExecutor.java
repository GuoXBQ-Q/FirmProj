package firmproj.utility;
import java.util.concurrent.*;

public class TimeoutTaskExecutor {

    private final ExecutorService executor;
    private final int timeout;

    public TimeoutTaskExecutor(int timeout) {
        this.timeout = timeout;
        this.executor = Executors.newSingleThreadExecutor();
    }

    public void executeWithTimeout(Runnable task) {
        Future<?> future = executor.submit(task);

        try {
            future.get(timeout, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            System.err.println("Task timed out.");
            future.cancel(true); // 取消任务
        } catch (InterruptedException | ExecutionException e) {
            System.err.println("Task execution failed: " + e.getMessage());
        }
    }

    public void shutdown() {
        executor.shutdown();
    }
}