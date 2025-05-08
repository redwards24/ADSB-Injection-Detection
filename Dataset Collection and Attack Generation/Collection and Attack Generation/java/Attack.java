
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import opensky.StateVectorIO;
import opensky.StateVectorList;

public class Attack {

    static String attack(final File file, final String type, final String folder) {
        final String path
                = Main.DATA_DIR
                + "\\attacked_sets\\"
                + folder + "\\"
                + file.getName().substring(0, file.getName().length() - 4)
                + "_" + type + ".csv";
        final ProcessBuilder builder = new ProcessBuilder(
                "python",
                ".\\python\\" + type + ".py",
                file.getAbsolutePath(),
                path
        );

        builder.redirectErrorStream(true);

        try {
            final Process process = builder.start();
            BufferedReader r = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while (true) {
                line = r.readLine();
                if (line == null) {
                    break;
                }
                System.out.println(line);
            }
            process.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return path + ";" + type;
    }

    public static void main(String[] args) {

        // Create Directories
        try {
            System.out.print("Creating directories for attack sets...");
            Files.createDirectories(Paths.get("data\\attacked_sets\\path_modification\\"));
            Files.createDirectories(Paths.get("data\\attacked_sets\\velocity_drift\\"));
            Files.createDirectories(Paths.get("data\\attacked_sets\\ghost_injection\\"));
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        System.out.println(" Success");

        final List<File> files = Arrays.asList(
                Path.of(Main.COMPLETE_SETS_PATH).toFile().listFiles());

        final ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        // Attack Each Airport
        files.forEach(file -> {
            System.out.println("Attacking: " + file.getName());
            es.submit(() -> {
                Arrays.asList(
                        attack(file, "PM", "path_modification"),
                        attack(file, "VD", "velocity_drift"),
                        attack(file, "GI", "ghost_injection")
                )
                .forEach(attack -> {
                    final String[] split = attack.split(";");

                    final StateVectorList data = StateVectorIO.read(split[0]).get();
                    // Create random sets
                    final List<StateVectorList> randLists = data.groupBy(sv -> new HourAirlineKey(sv.hour(), sv.getAirline()))
                            .values().stream()
                            .map(list -> list.getRandomList(list.size()))
                            .toList();
                    final StateVectorList rand = new StateVectorList(700);
                    boolean run = true;
                    int index = 0;
                    while (run) {
                        for (var list : randLists) {
                            if (rand.size() == 700) {
                                run = false;
                                break;
                            }
                            if (index < list.size()) {
                                rand.add(list.get(index));
                            }
                        }
                        ++index;
                    }

                    // Write random set to a file
                    StateVectorIO.write(rand, String.format(
                            "%s%s%s_%s_rand.csv",
                            Main.RANDOM_SETS_PATH, File.separator,
                            file.getName().substring(0, file.getName().length() - 4),
                            split[1]));
                });
            });
        });

        es.shutdown();
        try {
            System.out.print("Waiting for completion...");
            es.awaitTermination(60, TimeUnit.MINUTES);
            System.out.println(" all tasks completed.");
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // Should never enter this, but used just in case
        while (!es.isTerminated()) {
            // Wait
        }

    }

}
