using System.Net;

class Program
{
    static void Main()
    {
        Console.WriteLine($"\nBegin banknote feature engineering binary classification demo\n");
        // data
        string url = "https://raw.githubusercontent.com/grensen/id24/main/data/banknote_fixed.txt";
        var lines = new WebClient().DownloadString(url)
                                   .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                                   .Skip(1) // skip header
                                   .Select(l => l.Split(',').Select(float.Parse).ToArray())
                                   .ToArray();

        int n = lines.Length, f = lines[0].Length - 1;
        float[][] data = new float[n][];
        float[] labels = new float[n];
        float[] min = new float[f], max = new float[f];

        for (int j = 0; j < f; j++)
        {
            min[j] = lines.Min(r => r[j + 1]);
            max[j] = lines.Max(r => r[j + 1]);
        }

        for (int i = 0; i < n; i++)
        {
            labels[i] = lines[i][0];
            data[i] = new float[f];
            for (int j = 0; j < f; j++)
                data[i][j] = (lines[i][j + 1] - min[j]) / (max[j] - min[j]);
        }

        Console.WriteLine($"Download banknote dataset and normalize each feature between 0 and 1\n");


        // quick check
        for (int i = 0; i < 3; i++)
            Console.WriteLine($"X[{i}]: [{string.Join(", ", data[i])}], y: {labels[i]}");

        // build features
        float weight = 1.37f;
        Console.WriteLine($"\nPrediction weight: {weight} > F7 ? Forged(0) : Genuine(1)\n");

        float[] F5 = BuildFeature(data, d => d[0] + d[1] + d[2], "F5(Norm(F1+F2+F3))");
        float[] F6 = BuildFeature(data, d => d[0] * d[1] * d[2], "F6(Norm(F1*F2*F3))");
        float[] F7 = F5.Select((f5, i) => f5 * 3 + F6[i]).ToArray();
        Console.WriteLine($"F7(Norm(F5*3+F6)): min {F7.Min():F3}, max {F7.Max():F3}");
        
        // check model accuracy 
        int correct = 0;
        float totalError = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float prediction = F7[i];
            int predictedLabel = prediction > weight ? 0 : 1;

            if (predictedLabel == labels[i]) correct++;
            totalError += Math.Abs(predictedLabel - labels[i]);
        }

        float accuracy = (float)correct / data.Length;
        float mae = totalError / data.Length;

        Console.WriteLine($"\nAccuracy: {accuracy:P2}");
        Console.WriteLine($"MAE: {mae:F3}");


        Console.WriteLine($"\nFirst three F7 predictions:");
        for (int i = 0; i < 3; i++)
            Console.WriteLine($"X[{i}]F7: [{F7[i]:F4}]");

    }
    static float[] BuildFeature(float[][] data, Func<float[], float> selector, string name)
    {
        var feature = data.Select(selector).ToArray();
        var min = feature.Min();
        var max = feature.Max();
        Console.WriteLine($"{name}: min {min:F3}, max {max:F3}");
        return feature.Select(v => Norm(v, min, max)).ToArray();
    }

    static float Norm(float v, float mn, float mx) => (v - mn) / (mx - mn);
}

