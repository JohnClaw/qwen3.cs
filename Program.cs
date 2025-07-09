using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

// ---------- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ----------
file static class Globals
{
    public static int GS = 0;
}

// ---------- ОСНОВНЫЕ СТРУКТУРЫ ДАННЫХ ----------
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public record struct Config
{
    public int magic_number;
    public int version;
    public int dim;
    public int hidden_dim;
    public int n_layers;
    public int n_heads;
    public int n_kv_heads;
    public int vocab_size;
    public int seq_len;
    public int head_dim;
    public int shared_classifier; // bool
    public int group_size;

    public bool IsSharedClassifier => shared_classifier == 1;
}

public class QuantizedTensor
{
    public sbyte[] Q { get; }
    public float[] S { get; }

    public QuantizedTensor(sbyte[] q, float[] s)
    {
        Q = q;
        S = s;
    }
}

public class TransformerWeights
{
    public float[] RmsAttWeight { get; }
    public float[] RmsFfnWeight { get; }
    public float[] RmsFinalWeight { get; }
    public float[] QLnWeights { get; }
    public float[] KLnWeights { get; }

    public QuantizedTensor QTokens { get; }
    public QuantizedTensor[] Wq { get; }
    public QuantizedTensor[] Wk { get; }
    public QuantizedTensor[] Wv { get; }
    public QuantizedTensor[] Wo { get; }
    public QuantizedTensor[] W1 { get; }
    public QuantizedTensor[] W2 { get; }
    public QuantizedTensor[] W3 { get; }
    public QuantizedTensor Wcls { get; }

    public float[] TokenEmbeddingTable { get; }

    public TransformerWeights(Config p, ReadOnlyMemory<byte> memory)
    {
        var offset = 0;
        
        RmsAttWeight = ReadFloats(ref memory, ref offset, p.n_layers * p.dim);
        RmsFfnWeight = ReadFloats(ref memory, ref offset, p.n_layers * p.dim);
        RmsFinalWeight = ReadFloats(ref memory, ref offset, p.dim);
        QLnWeights = ReadFloats(ref memory, ref offset, p.n_layers * p.head_dim);
        KLnWeights = ReadFloats(ref memory, ref offset, p.n_layers * p.head_dim);

        QTokens = ReadQuantizedTensors(ref memory, ref offset, 1, p.vocab_size * p.dim)[0];
        Wq = ReadQuantizedTensors(ref memory, ref offset, p.n_layers, p.dim * (p.n_heads * p.head_dim));
        Wk = ReadQuantizedTensors(ref memory, ref offset, p.n_layers, p.dim * (p.n_kv_heads * p.head_dim));
        Wv = ReadQuantizedTensors(ref memory, ref offset, p.n_layers, p.dim * (p.n_kv_heads * p.head_dim));
        Wo = ReadQuantizedTensors(ref memory, ref offset, p.n_layers, (p.n_heads * p.head_dim) * p.dim);
        W1 = ReadQuantizedTensors(ref memory, ref offset, p.n_layers, p.dim * p.hidden_dim);
        W2 = ReadQuantizedTensors(ref memory, ref offset, p.n_layers, p.hidden_dim * p.dim);
        W3 = ReadQuantizedTensors(ref memory, ref offset, p.n_layers, p.dim * p.hidden_dim);

        Wcls = p.IsSharedClassifier
            ? QTokens
            : ReadQuantizedTensors(ref memory, ref offset, 1, p.dim * p.vocab_size)[0];

        TokenEmbeddingTable = new float[p.vocab_size * p.dim];
        Dequantize(QTokens, TokenEmbeddingTable.AsSpan());
    }

    private static float[] ReadFloats(ref ReadOnlyMemory<byte> memory, ref int offset, int count)
    {
        var slice = memory.Slice(offset, count * sizeof(float));
        offset += count * sizeof(float);
        var floatArray = new float[count];
        MemoryMarshal.Cast<byte, float>(slice.Span).CopyTo(floatArray);
        return floatArray;
    }

    private static QuantizedTensor[] ReadQuantizedTensors(ref ReadOnlyMemory<byte> memory, ref int offset, int numTensors, int tensorSize)
    {
        var tensors = new QuantizedTensor[numTensors];
        for (int i = 0; i < numTensors; i++)
        {
            var qMem = memory.Slice(offset, tensorSize);
            offset += tensorSize;

            int scalesSize = tensorSize / Globals.GS;
            var sMem = memory.Slice(offset, scalesSize * sizeof(float));
            offset += scalesSize * sizeof(float);

            var qArray = new sbyte[tensorSize];
            MemoryMarshal.Cast<byte, sbyte>(qMem.Span).CopyTo(qArray);
            
            var sArray = new float[scalesSize];
            MemoryMarshal.Cast<byte, float>(sMem.Span).CopyTo(sArray);

            tensors[i] = new QuantizedTensor(qArray, sArray);
        }
        return tensors;
    }
    
    private static void Dequantize(QuantizedTensor qx, Span<float> x)
    {
        var qSpan = qx.Q.AsSpan();
        var sSpan = qx.S.AsSpan();
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = qSpan[i] * sSpan[i / Globals.GS];
        }
    }
}

public class RunState
{
    public float[] X { get; }
    public float[] Xb { get; }
    public float[] Xb2 { get; }
    public float[] Hb { get; }
    public float[] Hb2 { get; }
    public sbyte[] Xq_q { get; }
    public float[] Xq_s { get; }
    public sbyte[] Hq_q { get; }
    public float[] Hq_s { get; }
    public float[] Q { get; }
    public float[] Att { get; }
    public float[] Logits { get; }
    public float[] KeyCache { get; }
    public float[] ValueCache { get; }

    public RunState(Config p)
    {
        int allHeadsDim = p.n_heads * p.head_dim;
        int kvDim = p.n_kv_heads * p.head_dim;

        X = new float[p.dim];
        Xb = new float[allHeadsDim];
        Xb2 = new float[p.dim];
        Hb = new float[p.hidden_dim];
        Hb2 = new float[p.hidden_dim];
        Xq_q = new sbyte[allHeadsDim];
        Xq_s = new float[allHeadsDim / Globals.GS];
        Hq_q = new sbyte[p.hidden_dim];
        Hq_s = new float[p.hidden_dim / Globals.GS];
        Q = new float[allHeadsDim];
        Att = new float[p.n_heads * p.seq_len];
        Logits = new float[p.vocab_size];
        KeyCache = new float[(long)p.n_layers * p.seq_len * kvDim];
        ValueCache = new float[(long)p.n_layers * p.seq_len * kvDim];
    }
}

// ---------- КЛАСС ТРАНСФОРМЕРА ----------
public class Transformer : IDisposable
{
    public Config Config { get; }
    private TransformerWeights Weights { get; }
    private RunState State { get; }
    private MemoryMappedFile? _mmf; 

    public Transformer(string checkpointPath, int ctxLength)
    {
        _mmf = MemoryMappedFile.CreateFromFile(checkpointPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        
        unsafe
        {
            byte* ptr = null;
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
            
            Config = MemoryMarshal.Read<Config>(new ReadOnlySpan<byte>(ptr, Unsafe.SizeOf<Config>()));
            
            if (Config.magic_number != 0x616a6331)
                throw new InvalidDataException($"File {checkpointPath} is not a qwen3.c checkpoint");
            if (Config.version != 1)
                throw new InvalidDataException($"Checkpoint {checkpointPath} is version {Config.version}, expected 1");
            
            if (ctxLength > 0 && ctxLength <= Config.seq_len)
            {
                Config = Config with { seq_len = ctxLength };
            }
            
            Globals.GS = Config.group_size;

            var weightsMemory = new ReadOnlySpan<byte>(ptr + 256, (int)(accessor.Capacity - 256)).ToArray();
            Weights = new TransformerWeights(Config, weightsMemory);

            accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        }
        
        State = new RunState(Config);
    }
    
    public void Dispose()
    {
        _mmf?.Dispose();
        _mmf = null;
        GC.SuppressFinalize(this);
    }

    public static void Softmax(Span<float> x)
    {
        if (x.IsEmpty) return;
        
        float maxVal = float.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] > maxVal) maxVal = x[i];
        }

        float sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = MathF.Exp(x[i] - maxVal);
            sum += x[i];
        }

        if (sum == 0) return;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] /= sum;
        }
    }
    
    private static void RmsNorm(Span<float> o, ReadOnlySpan<float> x, ReadOnlySpan<float> weight)
    {
        double ss = 0;
        foreach (float val in x)
        {
            ss += (double)val * val;
        }
        ss /= x.Length;
        ss = 1.0 / Math.Sqrt(ss + 1e-6);

        for (int j = 0; j < x.Length; j++)
        {
            o[j] = weight[j] * (float)(ss * x[j]);
        }
    }
    
    private static void Quantize(Span<sbyte> q, Span<float> s, ReadOnlySpan<float> x)
    {
        int n = x.Length;
        int numGroups = n / Globals.GS;
        const float qMax = 127.0f;

        for (int group = 0; group < numGroups; group++)
        {
            var xGroup = x.Slice(group * Globals.GS, Globals.GS);
            
            float wmax = 0;
            foreach (float val in xGroup)
            {
                float absVal = MathF.Abs(val);
                if (absVal > wmax) wmax = absVal;
            }
            
            float scale = wmax / qMax;
            if (scale == 0f) scale = 1.0f;
            
            s[group] = scale;

            var qGroup = q.Slice(group * Globals.GS, Globals.GS);
            for (int i = 0; i < Globals.GS; i++)
            {
                float quant_value = xGroup[i] / scale;
                qGroup[i] = (sbyte)Math.Round(quant_value);
            }
        }
    }

    private static void Matmul(float[] xout, sbyte[] x_q, float[] x_s, QuantizedTensor w, int n, int d)
    {
        var wQ = w.Q;
        var wS = w.S;
        
        Parallel.For(0, d, i =>
        {
            float val = 0;
            int in_offset = i * n;
            
            for (int j = 0; j <= n - Globals.GS; j += Globals.GS)
            {
                int ival = 0;
                for (int k = 0; k < Globals.GS; k++)
                {
                    ival += x_q[j + k] * wQ[in_offset + j + k];
                }
                val += ival * wS[(in_offset + j) / Globals.GS] * x_s[j / Globals.GS];
            }
            xout[i] = val;
        });
    }

    public float[] Forward(int token, int pos)
    {
        var p = Config;
        var w = Weights;
        var s = State;
        var x = s.X.AsSpan();
        int dim = p.dim;
        int kvDim = p.n_kv_heads * p.head_dim;
        int kvMul = p.n_heads / p.n_kv_heads;
        int hiddenDim = p.hidden_dim;
        int allHeadsDim = p.n_heads * p.head_dim;

        w.TokenEmbeddingTable.AsSpan(token * dim, dim).CopyTo(x);

        for (int l = 0; l < p.n_layers; l++)
        {
            long loff = (long)l * p.seq_len * kvDim;
            var k_current = s.KeyCache.AsSpan((int)(loff + (long)pos * kvDim), kvDim);
            var v_current = s.ValueCache.AsSpan((int)(loff + (long)pos * kvDim), kvDim);

            RmsNorm(s.Xb.AsSpan(0, dim), x, w.RmsAttWeight.AsSpan(l * dim, dim));
            
            Quantize(s.Xq_q.AsSpan(0, dim), s.Xq_s.AsSpan(0, dim / Globals.GS), s.Xb.AsSpan(0, dim));
            
            Matmul(s.Q, s.Xq_q, s.Xq_s, w.Wq[l], dim, allHeadsDim);
            Matmul(s.KeyCache, (int)(loff + (long)pos * kvDim), s.Xq_q, s.Xq_s, w.Wk[l], dim, kvDim);
            Matmul(s.ValueCache, (int)(loff + (long)pos * kvDim), s.Xq_q, s.Xq_s, w.Wv[l], dim, kvDim);

            var gq = w.QLnWeights.AsSpan(l * p.head_dim, p.head_dim);
            var gk = w.KLnWeights.AsSpan(l * p.head_dim, p.head_dim);

            for (int h = 0; h < p.n_heads; h++)
            {
                var q = s.Q.AsSpan(h * p.head_dim, p.head_dim);
                RmsNorm(q, q, gq);
                for (int j = 0; j < p.head_dim / 2; j++)
                {
                    float freq = MathF.Pow(1000000.0f, -(float)j / (p.head_dim / 2.0f));
                    float val = pos * freq;
                    (float sin_freq, float cos_freq) = MathF.SinCos(val);
                    float q_real = q[j];
                    float q_imag = q[j + p.head_dim / 2];
                    q[j] = q_real * cos_freq - q_imag * sin_freq;
                    q[j + p.head_dim / 2] = q_real * sin_freq + q_imag * cos_freq;
                }
            }

            for (int h = 0; h < p.n_kv_heads; h++)
            {
                var k_h = k_current.Slice(h * p.head_dim, p.head_dim);
                RmsNorm(k_h, k_h, gk);
                 for (int j = 0; j < p.head_dim / 2; j++)
                {
                    float freq = MathF.Pow(1000000.0f, -(float)j / (p.head_dim / 2.0f));
                    float val = pos * freq;
                    (float sin_freq, float cos_freq) = MathF.SinCos(val);
                    float k_real = k_h[j];
                    float k_imag = k_h[j + p.head_dim / 2];
                    k_h[j] = k_real * cos_freq - k_imag * sin_freq;
                    k_h[j + p.head_dim / 2] = k_real * sin_freq + k_imag * cos_freq;
                }
            }
            
            Parallel.For(0, p.n_heads, h =>
            {
                var q = s.Q.AsSpan(h * p.head_dim, p.head_dim);
                var att = s.Att.AsSpan(h * p.seq_len, p.seq_len);
                var keyCacheLayer = s.KeyCache.AsSpan((int)loff, p.seq_len * kvDim);

                for (int t = 0; t <= pos; t++)
                {
                    var k = keyCacheLayer.Slice(t * kvDim + (h / kvMul) * p.head_dim, p.head_dim);
                    double score = 0;
                    for (int i = 0; i < p.head_dim; i++) score += q[i] * k[i];
                    att[t] = (float)score / MathF.Sqrt(p.head_dim);
                }

                Softmax(att.Slice(0, pos + 1));
                
                var xb_h = s.Xb.AsSpan(h * p.head_dim, p.head_dim);
                xb_h.Clear();
                var valueCacheLayer = s.ValueCache.AsSpan((int)loff, p.seq_len * kvDim);

                for (int t = 0; t <= pos; t++)
                {
                    var v = valueCacheLayer.Slice(t * kvDim + (h / kvMul) * p.head_dim, p.head_dim);
                    float a = att[t];
                    for (int i = 0; i < p.head_dim; i++) xb_h[i] += a * v[i];
                }
            });
            
            Quantize(s.Xq_q.AsSpan(0, allHeadsDim), s.Xq_s.AsSpan(0, allHeadsDim / Globals.GS), s.Xb.AsSpan(0, allHeadsDim));
            Matmul(s.Xb2, s.Xq_q, s.Xq_s, w.Wo[l], allHeadsDim, dim);

            for (int i = 0; i < dim; i++) x[i] += s.Xb2[i];
            
            RmsNorm(s.Xb.AsSpan(0, dim), x, w.RmsFfnWeight.AsSpan(l * dim, dim));
            
            Quantize(s.Xq_q.AsSpan(0, dim), s.Xq_s.AsSpan(0, dim/Globals.GS), s.Xb.AsSpan(0, dim));
            
            Matmul(s.Hb, s.Xq_q, s.Xq_s, w.W1[l], dim, hiddenDim);
            Matmul(s.Hb2, s.Xq_q, s.Xq_s, w.W3[l], dim, hiddenDim);
            
            var hbSpan = s.Hb.AsSpan();
            var hb2Span = s.Hb2.AsSpan();
            for (int i = 0; i < hiddenDim; i++)
            {
                float val = hbSpan[i];
                val *= 1.0f / (1.0f + MathF.Exp(-val));
                val *= hb2Span[i];
                hbSpan[i] = val;
            }
            
            Quantize(s.Hq_q, s.Hq_s, hbSpan);
            Matmul(s.Xb, s.Hq_q, s.Hq_s, w.W2[l], hiddenDim, dim);
            
            for (int i = 0; i < dim; i++) x[i] += s.Xb[i];
        }

        RmsNorm(x, x, w.RmsFinalWeight);
        
        Quantize(s.Xq_q.AsSpan(0, dim), s.Xq_s.AsSpan(0, dim/Globals.GS), x);
        Matmul(s.Logits, s.Xq_q, s.Xq_s, w.Wcls, dim, p.vocab_size);

        return s.Logits;
    }
    
    private static void Matmul(float[] xout, int xout_offset, sbyte[] x_q, float[] x_s, QuantizedTensor w, int n, int d)
    {
        var wQ = w.Q;
        var wS = w.S;
        
        Parallel.For(0, d, i =>
        {
            float val = 0;
            int in_offset = i * n;
            
            for (int j = 0; j <= n - Globals.GS; j += Globals.GS)
            {
                int ival = 0;
                for (int k = 0; k < Globals.GS; k++)
                {
                    ival += x_q[j + k] * wQ[in_offset + j + k];
                }
                val += ival * wS[(in_offset + j) / Globals.GS] * x_s[j / Globals.GS];
            }
            xout[xout_offset + i] = val;
        });
    }
}


// ---------- КЛАСС ТОКЕНИЗАТОРА ----------
public class Tokenizer
{
    public int VocabSize { get; }
    public uint BosTokenId { get; }
    public uint EosTokenId { get; }
    public string PromptTemplate { get; }
    public string SystemPromptTemplate { get; }
    private readonly string[] _vocab;
    private readonly float[] _mergeScores;
    private readonly Dictionary<string, int> _vocabDict;
    
    public Tokenizer(string checkpointPath, int configVocabSize, bool enableThinking)
    {
        string tokenizerPath = $"{checkpointPath}.tokenizer";
        if (!File.Exists(tokenizerPath)) throw new FileNotFoundException("Tokenizer file not found", tokenizerPath);
        
        var tempVocab = new List<string>();
        var tempScores = new List<float>();

        using var reader = new BinaryReader(File.OpenRead(tokenizerPath));
        
        reader.ReadUInt32(); // max_token_length, not used
        BosTokenId = reader.ReadUInt32();
        EosTokenId = reader.ReadUInt32();

        int id_counter = 0;
        while(reader.BaseStream.Position < reader.BaseStream.Length)
        {
            // Эта логика чтения теперь точно соответствует C-коду, который может иметь токены без score.
            float score = 0;
            if (reader.BaseStream.Position + sizeof(float) <= reader.BaseStream.Length)
            {
                score = reader.ReadSingle();
            }

            int len = 0;
            if (reader.BaseStream.Position + sizeof(int) <= reader.BaseStream.Length)
            {
                 len = reader.ReadInt32();
            }

            if (len > 0)
            {
                 if (reader.BaseStream.Position + len > reader.BaseStream.Length) break;
                 string tokenStr = Encoding.UTF8.GetString(reader.ReadBytes(len));
                 tempVocab.Add(tokenStr);
            }
            else
            {
                // Токен может быть пустой строкой, как в C-коде
                tempVocab.Add("");
            }
            tempScores.Add(score);
            id_counter++;
        }
        
        VocabSize = tempVocab.Count;
        _vocab = tempVocab.ToArray();
        _mergeScores = tempScores.ToArray();
        
        // Заполняем словарь для быстрого поиска, он нужен в Encode
        _vocabDict = new Dictionary<string, int>();
        for(int i = 0; i < VocabSize; i++)
        {
            if(!string.IsNullOrEmpty(_vocab[i]))
            {
                _vocabDict[_vocab[i]] = i;
            }
        }
        
        if(VocabSize != configVocabSize)
        {
            Console.WriteLine($"Warning: vocab_size in config ({configVocabSize}) does not match the actual number of tokens in tokenizer ({VocabSize}).");
        }

        PromptTemplate = LoadPromptTemplate(checkpointPath, withSystemPrompt: false, enableThinking: enableThinking);
        SystemPromptTemplate = LoadPromptTemplate(checkpointPath, withSystemPrompt: true, enableThinking: enableThinking);
    }

    private string LoadPromptTemplate(string checkpointPath, bool withSystemPrompt, bool enableThinking)
    {
        var suffix = withSystemPrompt
            ? (enableThinking ? ".template.with-system-and-thinking" : ".template.with-system")
            : (enableThinking ? ".template.with-thinking" : ".template");
        
        var path = $"{checkpointPath}{suffix}";
        if (!File.Exists(path)) throw new FileNotFoundException($"Could not load prompt template: {path}");
        return File.ReadAllText(path).Replace("\r\n", "\n");
    }
    
    public string Decode(int token) => (token >= 0 && token < _vocab.Length) ? _vocab[token] : "";

    // --- ИСПРАВЛЕНИЕ: Полностью переписанный метод Encode ---
    // Эта реализация теперь точно повторяет логику C-кода, включая обработку
    // специальных токенов `<...>` и побайтовую обработку остального текста.
    public int[] Encode(string text)
    {
        var tokens = new List<int>();
        var utf8Bytes = Encoding.UTF8.GetBytes(text);
        var tempBuffer = new byte[2]; // Для одного UTF-8 символа + null

        for (int i = 0; i < utf8Bytes.Length; i++)
        {
            int id = -1;
            bool foundSpecialToken = false;

            if (utf8Bytes[i] == (byte)'<')
            {
                int endTokenPos = -1;
                for (int k = i; k < utf8Bytes.Length && k < i + 64; k++)
                {
                    if (utf8Bytes[k] == (byte)'>')
                    {
                        endTokenPos = k;
                        break;
                    }
                }

                if (endTokenPos != -1)
                {
                    var specialTokenBytes = new Span<byte>(utf8Bytes, i, endTokenPos - i + 1);
                    string specialTokenStr = Encoding.UTF8.GetString(specialTokenBytes);
                    if (_vocabDict.TryGetValue(specialTokenStr, out id))
                    {
                        tokens.Add(id);
                        i = endTokenPos; // Перемещаем главный указатель
                        foundSpecialToken = true;
                    }
                }
            }
            
            if (!foundSpecialToken)
            {
                tempBuffer[0] = utf8Bytes[i];
                string singleByteStr = Encoding.UTF8.GetString(tempBuffer, 0, 1);
                if (_vocabDict.TryGetValue(singleByteStr, out id))
                {
                    tokens.Add(id);
                }
                else
                {
                    // В С-коде здесь была ошибка, которая добавляла мусор.
                    // Мы будем просто пропускать неизвестные символы, что безопаснее.
                    Console.WriteLine($"Warning: unknown character byte {utf8Bytes[i]} in input, skipping.");
                }
            }
        }

        // Цикл слияния BPE (остается без изменений, он был корректен)
        while (tokens.Count >= 2)
        {
            float best_score = -1e10f;
            int best_idx = -1;
            int best_id = -1;
            
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                string merged = _vocab[tokens[i]] + _vocab[tokens[i + 1]];
                if (_vocabDict.TryGetValue(merged, out int id) && _mergeScores[id] > best_score)
                {
                    best_score = _mergeScores[id];
                    best_idx = i;
                    best_id = id;
                }
            }

            if (best_idx == -1)
            {
                break;
            }

            tokens[best_idx] = best_id;
            tokens.RemoveAt(best_idx + 1);
        }
        return tokens.ToArray();
    }
}


// ---------- КЛАСС СЭМПЛЕРА ----------
public class Sampler
{
    private readonly int _vocabSize;
    private readonly float _temperature;
    private readonly float _topp;
    private ulong _rngState;
    private readonly ProbIndex[] _probIndex;
    
    private struct ProbIndex { public float Prob; public int Index; }
    
    public Sampler(int vocabSize, float temperature, float topp, ulong rngSeed)
    {
        _vocabSize = vocabSize; // Здесь должен быть РЕАЛЬНЫЙ размер словаря
        _temperature = temperature;
        _topp = topp;
        _rngState = rngSeed;
        _probIndex = new ProbIndex[vocabSize];
    }
    
    private uint RandomU32()
    {
        _rngState ^= _rngState >> 12;
        _rngState ^= _rngState << 25;
        _rngState ^= _rngState >> 27;
        return (uint)((_rngState * 0x2545F4914F6CDD1DUL) >> 32);
    }
    
    private float RandomF32() => (RandomU32() >> 8) / 16777216.0f;

    private int SampleArgmax(ReadOnlySpan<float> probabilities)
    {
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++)
        {
            if (probabilities[i] > max_p)
            {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        return max_i;
    }

    private int SampleMult(ReadOnlySpan<float> probabilities, float coin)
    {
        float cdf = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cdf += probabilities[i];
            if (coin < cdf) return i;
        }
        return probabilities.Length - 1;
    }
    
    private int SampleTopp(ReadOnlySpan<float> probabilities, float coin)
    {
        var probIndex = _probIndex.AsSpan(0, _vocabSize); // Используем реальный vocabSize
        int n0 = 0;
        
        float cutoff = (1.0f - _topp) / (_vocabSize - 1);
        for (int i = 0; i < _vocabSize; i++)
        {
            if (probabilities[i] >= cutoff)
            {
                probIndex[n0].Index = i;
                probIndex[n0].Prob = probabilities[i];
                n0++;
            }
        }
        
        probIndex.Slice(0, n0).Sort((a, b) => b.Prob.CompareTo(a.Prob));
        
        float cumulativeProb = 0;
        int lastIdx = n0 - 1;
        for (int i = 0; i < n0; i++)
        {
            cumulativeProb += probIndex[i].Prob;
            if (cumulativeProb > _topp)
            {
                lastIdx = i;
                break;
            }
        }
        
        float r = coin * cumulativeProb;
        float cdf = 0;
        for (int i = 0; i <= lastIdx; i++)
        {
            cdf += probIndex[i].Prob;
            if (r < cdf) return probIndex[i].Index;
        }
        return probIndex[lastIdx].Index;
    }
    
    public int Sample(Span<float> logits)
    {
        // Теперь метод не принимает vocabSize, он использует тот, с которым был создан
        var relevantLogits = logits.Slice(0, _vocabSize);

        if (_temperature == 0.0f)
        {
            return SampleArgmax(relevantLogits);
        }
        
        for (int q = 0; q < relevantLogits.Length; q++) relevantLogits[q] /= _temperature;
        
        Transformer.Softmax(relevantLogits);
        
        float coin = RandomF32();

        if (_topp <= 0 || _topp >= 1)
        {
            return SampleMult(relevantLogits, coin);
        }
        else
        {
            return SampleTopp(relevantLogits, coin);
        }
    }
}


// ---------- ГЛАВНАЯ ПРОГРАММА И ЦИКЛЫ ГЕНЕРАЦИИ ----------
class Program
{
    static void Generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler, string prompt)
    {
        var promptTokens = tokenizer.Encode(prompt);
        int numPromptTokens = promptTokens.Length;
        if (numPromptTokens < 1)
        {
            Console.Error.WriteLine("Cannot encode prompt.");
            return;
        }

        int token = promptTokens[0];
        int pos = 0;
        var sw = Stopwatch.StartNew();

        while (pos < transformer.Config.seq_len)
        {
            float[] logits = transformer.Forward(token, pos);
            int next;
            if (pos < numPromptTokens - 1)
            {
                next = promptTokens[pos + 1];
            }
            else
            {
                next = sampler.Sample(logits);
            }
            
            Console.Write(tokenizer.Decode(token));

            pos++;
            token = next;

            if (pos >= numPromptTokens && (token == tokenizer.EosTokenId || token == tokenizer.BosTokenId))
            {
                break;
            }
        }
        sw.Stop();
        Console.WriteLine($"\n\nAchieved: {(pos > 1 ? (pos - 1) / sw.Elapsed.TotalSeconds : 0):F2} tokens/sec");
    }

    static void Chat(Transformer transformer, Tokenizer tokenizer, Sampler sampler, string? cliUserPrompt, string? systemPrompt)
    {
        int pos = 0;
        bool userTurn = true;
        int[] promptTokens = Array.Empty<int>();
        int userIdx = 0;
        int token = 0;
        int next_token = -1;
        
        while (pos < transformer.Config.seq_len)
        {
            if (userTurn)
            {
                if (pos >= transformer.Config.seq_len) {
                    Console.WriteLine("\n(context window full, clearing)");
                    pos = 0;
                }
                
                string userPrompt;
                if (cliUserPrompt != null && pos == 0) {
                    userPrompt = cliUserPrompt;
                    cliUserPrompt = null;
                } else {
                    Console.Write("\n> ");
                    userPrompt = Console.ReadLine() ?? "";
                    if (string.IsNullOrWhiteSpace(userPrompt)) break;
                }

                string renderedPrompt;
                if (pos == 0 && !string.IsNullOrEmpty(systemPrompt)) {
                    string template = tokenizer.SystemPromptTemplate;
                    int placeholderIndex = template.IndexOf("%s");
                    if (placeholderIndex != -1)
                    {
                        template = template.Remove(placeholderIndex, 2).Insert(placeholderIndex, systemPrompt);
                    }
                    renderedPrompt = template.Replace("%s", userPrompt);
                } else {
                    renderedPrompt = tokenizer.PromptTemplate.Replace("%s", userPrompt, StringComparison.Ordinal);
                }
                
                promptTokens = tokenizer.Encode(renderedPrompt);
                userIdx = 0;
                userTurn = false;
                Console.Write("< ");
            }

            if (userIdx < promptTokens.Length)
            {
                token = promptTokens[userIdx++];
            }
            else
            {
                token = next_token;
            }

            if (pos >= transformer.Config.seq_len) break;
            
            float[] logits = transformer.Forward(token, pos);
            next_token = sampler.Sample(logits);
            pos++;

            if (userIdx >= promptTokens.Length)
            {
                if (next_token == tokenizer.EosTokenId || next_token == tokenizer.BosTokenId)
                {
                    userTurn = true;
                }
                else
                {
                    Console.Write(tokenizer.Decode(next_token));
                }
            }
        }
    }
    
    static void Main(string[] args)
    {
        string? checkpointPath = null;
        float temperature = 1.0f;
        float topp = 0.9f;
        string? prompt = null;
        ulong rngSeed = 0;
        string mode = "chat";
        string? systemPrompt = null;
        bool enableThinking = false;
        int ctxLength = 0;
        
        if (args.Length < 1) { ErrorUsage(); return; }
        checkpointPath = args[0];

        for (int i = 1; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length || !args[i].StartsWith("-") || args[i].Length != 2) { ErrorUsage(); return; }
            string flag = args[i];
            string value = args[i + 1];
            switch (flag[1])
            {
                case 't': temperature = float.Parse(value, System.Globalization.CultureInfo.InvariantCulture); break;
                case 'p': topp = float.Parse(value, System.Globalization.CultureInfo.InvariantCulture); break;
                case 's': rngSeed = ulong.Parse(value); break;
                case 'c': ctxLength = int.Parse(value); break;
                case 'i': prompt = value; break;
                case 'm': mode = value; break;
                case 'y': systemPrompt = value; break;
                case 'r': enableThinking = int.Parse(value) == 1; break;
                default: ErrorUsage(); return;
            }
        }
        
        if (rngSeed == 0) rngSeed = (ulong)DateTime.Now.Ticks;
        if (temperature < 0) temperature = 0;
        if (topp < 0 || topp > 1) topp = 0.9f;
        
        try
        {
            using var transformer = new Transformer(checkpointPath, ctxLength);
            var tokenizer = new Tokenizer(checkpointPath, transformer.Config.vocab_size, enableThinking);
            
            // --- ИСПРАВЛЕНИЕ: Инициализация Sampler ---
            // Семплер должен быть инициализирован РЕАЛЬНЫМ размером словаря из токенизатора,
            // а не "дополненным" размером из конфига модели.
            var sampler = new Sampler(tokenizer.VocabSize, temperature, topp, rngSeed);

            if (string.IsNullOrEmpty(prompt))
            {
                Console.WriteLine($"hidden_size={transformer.Config.dim}, intermediate_size={transformer.Config.hidden_dim}, num_hidden_layers={transformer.Config.n_layers}, num_attention_heads={transformer.Config.n_heads}, num_kv_heads={transformer.Config.n_kv_heads}, head_dim={transformer.Config.head_dim}, ctx_length={transformer.Config.seq_len}, vocab_size={tokenizer.VocabSize}, shared_classifier={transformer.Config.shared_classifier}, quantization_block_size={transformer.Config.group_size}");
            }
            
            if (mode == "generate")
            {
                Generate(transformer, tokenizer, sampler, prompt ?? "");
            }
            else if (mode == "chat")
            {
                Chat(transformer, tokenizer, sampler, prompt, systemPrompt);
            }
            else
            {
                Console.Error.WriteLine($"Unknown mode: {mode}");
                ErrorUsage();
            }
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"An error occurred: {e.Message}");
            Console.Error.WriteLine(e.StackTrace);
        }
    }
    
    static void ErrorUsage()
    {
        Console.Error.WriteLine("Usage:   dotnet run -- <checkpoint> [options]");
        Console.Error.WriteLine("Example: dotnet run -- Qwen3-4B.bin -r 1");
        Console.Error.WriteLine("Options:");
        Console.Error.WriteLine("  -t <float>  temperature in [0,inf], default 1.0");
        Console.Error.WriteLine("  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9");
        Console.Error.WriteLine("  -s <int>    random seed, default time(NULL)");
        Console.Error.WriteLine("  -c <int>    context window size, 0 (default) = max_seq_len");
        Console.Error.WriteLine("  -m <string> mode: generate|chat, default: chat");
        Console.Error.WriteLine("  -i <string> input prompt");
        Console.Error.WriteLine("  -y <string> system prompt in chat mode, default is none");
        Console.Error.WriteLine("  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking");
    }
}