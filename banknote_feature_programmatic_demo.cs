// https://github.com/grensen/feature_engineering
// https://jamesmccaffrey.wordpress.com/2020/08/18/in-the-banknote-authentication-dataset-class-0-is-genuine-authentic/

using System.Net;
using System.Text.RegularExpressions;

class Program
{
    static void Main()
    {

        Console.WriteLine($"\nBegin banknote feature engineering on the fly demo\n");
        Console.WriteLine($"Download banknote dataset and keep original values\n");

        string url = "https://raw.githubusercontent.com/Saswat7101/Bank-Note-Authentication/refs/heads/main/BankNote_Authentication.csv";
        var lines = new WebClient().DownloadString(url)
                                    .Split('\n', StringSplitOptions.RemoveEmptyEntries)
                                    .Skip(1) // skip header
                                    .Select(l => l.Split(',').Select(float.Parse).ToArray())
                                    .ToArray();

        // split labels and data points
        int n = lines.Length, f = lines[0].Length - 1;
        float[][] data = new float[n][];
        float[] labels = new float[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = lines[i][^1];
            data[i] = new float[f];
            for (int j = 0; j < f; j++)
                data[i][j] = lines[i][j];
        }
        Console.WriteLine($"F1: variance, F2: skewness, F3: curtosis, F4: entropy");
        // show samples
        for (int i = 0; i < 2; i++)
            Console.WriteLine($"X[{i}]:  [{string.Join(", ", data[i].Select(x => x.ToString("F3")))}], y: {labels[i]}");
        Console.WriteLine($"X[{^1}]: [{string.Join(", ", data[^1].Select(x => x.ToString("F3")))}], y: {labels[^1]}");

        // extends the old features with3 new features: F5, F6, F7
        // F1+F2+F3=F5,F1*F2*F3=F6,F5+F5+F5+F6=F7
        string input = "F1+F2+F3,F1*F2*F3,F5+F5+F5+F6";
        Console.WriteLine($"\nCreate new features: F5, F6, F7");
        Console.WriteLine($"Feature-engineering input: {input}\n");

        bool useProgrammatic = true;
        bool showFeatEngProcess = true;

        FeatureEngineering fe = new(input, showFeatEngProcess);

        fe.Train(data);

        // prediction weights: predefined min/max for var, ske, cur, F5 and F6 from visual training dataset
        const float varianceMin = -7.0f, varianceMax = 6.8f; // F1
        const float skewnessMin = -13.8f, skewnessMax = 13.0f; // F2
        const float curtosisMin = -5.3f, curtosisMax = 17.9f; // F3
        const float minf5 = 1.02f, maxf5 = 1.9f; // F5
        const float minf6 = 0f, maxf6 = 0.16f; // F6
    
         // prediction model: on the fly prediction
        float[] F7 = new float[data.Length];

        float predictionWeight = 1.37f; // prediction weight from visual training with feature engineering
        Console.WriteLine($"Prediction weight: F7 > {predictionWeight} ? Genuine(0) : Forged(1)");
        
        // feature engineering equation
        for (int i = 0; i < data.Length; i++)
        {
            float v1 = (data[i][0] - varianceMin) / (varianceMax - varianceMin);
            float v2 = (data[i][1] - skewnessMin) / (skewnessMax - skewnessMin);
            float v3 = (data[i][2] - curtosisMin) / (curtosisMax - curtosisMin);

            // 1. Norm(F1 + F2 + F3)
            float f5 = v1 + v2 + v3;
            float norm5 = (f5 - minf5) / (maxf5 - minf5);
            
            // 2. Norm(F1 * F2 * F3)
            float f6 = v1 * v2 * v3;
            float norm6 = (f6 - minf6) / (maxf6 - minf6);

            // 3. Identity(F5 + F5 + F5 + F6)
            F7[i] = 3f * norm5 + norm6;
        }
        
        // check model accuracy 
        int correct = 0;
        float totalError = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float prediction = useProgrammatic ? fe.Predict(data[i]) : F7[i];

            int predictedLabel = prediction > predictionWeight ? 0 : 1;

            if (predictedLabel == labels[i]) correct++;
            totalError += Math.Abs(predictedLabel - labels[i]);
        }

        Console.WriteLine($"\nAccuracy: {(float)correct / data.Length:P2}");
        Console.WriteLine($"MAE: {totalError / data.Length:F3}");

        Console.WriteLine($"\nF7 predictions:");
        for (int i = 0; i < 2; i++)
            Console.WriteLine($"X[{i}]F7:  [{F7[i]:F3}], pred: {(F7[i] > predictionWeight ? 0 : 1)}");
        Console.WriteLine($"X[{^1}]F7: [{F7[^1]:F3}], pred: {(F7[^1] > predictionWeight ? 0 : 1)}");
        Console.WriteLine($"\nX transformations: Norm(F1, F2, F3, F4, F5, F6, F7)");
        for (int i = 0; i < 2; i++)
            Console.WriteLine($"X[{i}]: [{string.Join(", ", fe.Transform(data[i], true).Select(x => x.ToString("F3")))}]");
        Console.WriteLine($"X[{^1}]: [{string.Join(", ", fe.Transform(data[^1], true).Select(x => x.ToString("F3")))}]");
    }
}

class FeatureEngineering
{
    public float[] min, max;
    public int[][] featIds, featOperations;
    public string input;

    public void Train(float[][] _data)
    {
        // construction process to build the feature engineering pipeline
        float[][] data = _data.Select(innerArray => innerArray.ToArray()).ToArray();

        int f = data[0].Length;
        min = new float[data[0].Length];
        max = new float[data[0].Length];
        for (int j = 0; j < min.Length; j++)
        {
            min[j] = data.Min(r => r[j]);
            max[j] = data.Max(r => r[j]);
        }

        // normalization of the source dataset
        for (int i = 0; i < data.Length; i++)
            for (int j = 0; j < f; j++)
                data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);

        for (int i = 0; i < featOperations.Length; i++)
        {
            // 1. init new feature and feed first
            var fIds = featIds[i];
            float[] newFeat = new float[data.Length];

            for (int id = 0; id < data.Length; id++)
                newFeat[id] = data[id][fIds[0]];

            // 2. apply operators
            var operatorCodes = featOperations[i];
            for (int j = 0; j < operatorCodes.Length; j++)
            {
                int curFeat = fIds[j + 1];
                int operatorID = operatorCodes[j];
                for (int id = 0; id < newFeat.Length; id++)
                    newFeat[id] = ApplyOperation(newFeat[id], data[id][curFeat], operatorID);
            }

            // 3. get min max of new feature
            float mn = newFeat.Min(), mx = newFeat.Max();
            min = [.. min, mn];
            max = [.. max, mx];

            // 4. normalize new feature
            if (i < featOperations.Length - 1)
                for (int j = 0; j < newFeat.Length; j++)
                    newFeat[j] = Norm(newFeat[j], mn, mx);

            // 5. add new feature and simulate for final data+newFeats array
            float[][] data2 = new float[newFeat.Length][];
            for (int j = 0; j < newFeat.Length; j++)
                data2[j] = [.. data[j], newFeat[j]];
            data = data2;
        }
    }
    public float[] Transform(float[] _x, bool norm = false)
    {
        var x = _x;
        int f = x.Length;
        for (int j = 0; j < featIds.Length; j++)
        {
            float newFeat = 0;
            var operatorCodes = featOperations[j];
            var fIds = featIds[j];
            int fID = fIds[0];
            if (j < featOperations.Length - 1)
            {
                newFeat = Norm(x[fID], min[fID], max[fID]);
                for (int k = 1; k < fIds.Length; k++)
                {
                    fID = fIds[k];
                    newFeat = ApplyOperation(newFeat, Norm(x[fID], min[fID], max[fID]), operatorCodes[k - 1]);
                }
                newFeat = Norm(newFeat, min[f + j], max[f + j]);
            }
            else
            {
                newFeat = x[fID];
                for (int k = 1; k < fIds.Length; k++)
                {
                    fID = fIds[k];
                    newFeat = ApplyOperation(newFeat, x[fID], operatorCodes[k - 1]);
                }
                if (norm)
                    newFeat = Norm(newFeat, min[f + j], max[f + j]);
            }
            x = [.. x, newFeat];
        }
        if (norm)
            for (int i = 0; i < f; i++)
                x[i] = Norm(x[i], min[i], max[i]);

        return x;
    }

    public float Predict(float[] _x) => Transform(_x)[^1];
    
    public FeatureEngineering(string _input, bool info, float[][] _data = null)
    {

        // F1+F2+F3=F5,F1*F2*F3=F6,F5+F5+F5+F6=F7
        //Console.WriteLine($"\nFeature-engineering input: {input}\n");

        // input process: extracedFeatures + operators + min max
        input = _input;
        string[] parts = _input.Split(',');
        featIds = new int[parts.Length][];
        featOperations = new int[parts.Length][];

        for (int i = 0; i < parts.Length; i++)
        {
            if (info) Console.WriteLine($"Extracted part[{i}]: {parts[i]}");

            var featMatches = Regex.Matches(parts[i], @"F(\d+)");
            int[] fIds = new int[featMatches.Count];
            for (int j = 0; j < featMatches.Count; j++)
                fIds[j] = int.Parse(featMatches[j].Groups[1].Value) - 1;
            if (info) Console.WriteLine("Extracted features: " + string.Join(", ", fIds));
            featIds[i] = fIds;
            var operatorMatches = Regex.Matches(parts[i], @"([+\-*=])");
            char[] operators = new char[operatorMatches.Count];
            for (int j = 0; j < operatorMatches.Count; j++)
                operators[j] = operatorMatches[j].Value[0];

            int[] operatorCodes = new int[operatorMatches.Count];
            for (int j = 0; j < operatorMatches.Count; j++)
                operatorCodes[j] = ConvertOperatorToCode(operatorMatches[j].Value[0]);

            if (info) Console.WriteLine($"Extracted operators: {string.Join(", ", operatorCodes)}\n");
            featOperations[i] = operatorCodes; // stack operators each new feature
        }

        if (_data == null) return;

        Train(_data);

        if(info)
            for (int i = 0; i < min.Length; i++)
                Console.WriteLine($"Class FT{i + 1} min: {min[i],6:F2}, max: {max[i],6:F2}");
    }

    static float Norm(float v, float mn, float mx) => (v - mn) / (mx - mn);
    int ConvertOperatorToCode(char op)
    {
        return op switch
        {
            '+' => 0,   // Addition
            '-' => 1,   // Subtraction
            '*' => 2,   // Multiplication
            '=' => 3,   // Equality
            _ => 0      // Default to addition for unknown operators
        };
    }
    static float ApplyOperation(float currentValue, float newValue, int operatorID)
    {
        switch (operatorID)
        {
            case 0: return currentValue + newValue;
            case 1: return currentValue - newValue;
            case 2: return currentValue * newValue;
            default: return currentValue;
        }
    }
}