import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class CompareFiles {
    public static void main(String[] args) {
        try {
            BufferedReader br1 = new BufferedReader(new FileReader("hand_in.txt"));
            BufferedReader br2 = new BufferedReader(new FileReader("truth.txt"));
            String line1 = br1.readLine();
            String line2 = br2.readLine();
            float count1 = 0;
            float count2 = 0;
            while(line1 != null) {
                if(line1.equals(line2)) {
                    count1++;
                }
                count2++;
                line1 = br1.readLine();
                line2 = br2.readLine();
            }
            System.out.println("Procent rätt:  " + (count1 * 100 / count2));
            System.out.println(count2);
            
        } catch (IOException e) {
            System.err.println("Error");
        }
    }
}
