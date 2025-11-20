import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";
import path from "path";
import fs from "fs";
import Papa from "papaparse";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const PORT = 8100; // 若和 insight.js 同時執行就不能同 port

// 讓 __dirname 可用（ESM）
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 你的專案 ROOT（已根據你的資料夾結構確認）
const ROOT = path.resolve(__dirname, "..");

// 工具：讀 CSV
function readCSV(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const content = fs.readFileSync(filePath, "utf8");
  return Papa.parse(content, { header: true }).data;
}

// 工具：簡單壓縮資料，避免 prompt 爆掉
function sampleRows(rows, max = 40) {
  if (!Array.isArray(rows)) return [];
  return rows.slice(0, max);
}

// 工具：從 LLM 回傳文字中擷取 json 區塊
function extractJson(text) {
  if (!text) return null;
  const block = text.match(/```json([\s\S]*?)```/i);
  if (block) return block[1].trim();
  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first !== -1 && last !== -1 && last > first) {
    return text.slice(first, last + 1);
  }
  return null;
}

//==============================
//   /api/future-index
//==============================
app.get("/api/future-index", async (req, res) => {
  try {
    //-----------------------
    // ① 技術趨勢資料
    //-----------------------
    const techFiles = ["GaN.csv", "magsafe.csv", "pd.csv", "typeC.csv"];
    const techData = techFiles.map((file) => ({
      category: file.replace(".csv", ""),
      rows: sampleRows(readCSV(`${ROOT}/timeseries_TechTrends/${file}`), 30),
    }));

    //-----------------------
    // ② 使用者需求趨勢
    //-----------------------
    const userRequirementsFiles = [
      "powerBank_chargingSpeed.csv",
      "powerBank_compatibility.csv",
      "powerBank_design.csv",
      "powerBank_price.csv",
      "powerBank_weight.csv",
      "week_powerBank_chargingSpeed.csv",
      "week_powerBank_compatibility.csv",
      "week_powerBank_design.csv",
      "week_powerBank_price.csv",
      "week_powerBank_weight.csv",
    ];

    const userRequirementsData = userRequirementsFiles.map((file) => ({
      category: file.replace(".csv", ""),
      rows: sampleRows(
        readCSV(`${ROOT}/timeseries_UserRequirements/${file}`),
        30
      ),
    }));

    //-----------------------
    // ③ NLP / ABSA
    //-----------------------
    const absaAll = sampleRows(
      readCSV(`${ROOT}/nlp_result/absa_results.csv`),
      80
    );

    const pchomeAbsaFiles = [
      "improved_absa_results_500.csv",
      "improved_absa_results_1000.csv",
      "improved_absa_results_1500.csv",
      "improved_absa_results_2000.csv",
      "improved_absa_results_2000_10000.csv",
      "improved_absa_results_10000.csv",
    ];

    const pchomeAbsaData = pchomeAbsaFiles.map((file) => ({
      file: file.replace(".csv", ""),
      rows: sampleRows(readCSV(`${ROOT}/nlp_result/pchome/${file}`), 40),
    }));

    const pchomeSentimentDistFiles = [
      "sentiment_distribution_500.csv",
      "sentiment_distribution_1000.csv",
      "sentiment_distribution_1500.csv",
      "sentiment_distribution_2000.csv",
      "sentiment_distribution_2000_10000.csv",
      "sentiment_distribution_10000.csv",
    ];

    const pchomeSentimentDistData = pchomeSentimentDistFiles.map((file) => ({
      file: file.replace(".csv", ""),
      rows: sampleRows(
        readCSV(`${ROOT}/nlp_result/pchome/${file}`),
        50
      ),
    }));

    //-----------------------
    // ④ 爬蟲價格帶評論（成本效益分析重點）
    //-----------------------
    const reviewFiles = [
      "500.csv",
      "1000.csv",
      "1500.csv",
      "2000.csv",
      "2000_10000.csv",
      "10000.csv",
    ];

    const reviewData = reviewFiles.map((file) => ({
      priceRange: file.replace(".csv", ""),
      rows: sampleRows(readCSV(`${ROOT}/crawlers_result/${file}`), 40),
    }));

    //-----------------------
    // ⑤ 我方新品（假定）
    //-----------------------
    const fusion20Spec = {
      name: "Fusion-20",
      brand: "PowerPulse",
      capacity_mAh: 20000,
      maxWatt: 65,
      technology: ["GaN", "PD 3.1", "Qi2"],
      features: ["自帶 Type-C 線", "雙 C + A", "支援 MacBook 快充", "磁吸 15W"],
      targetPriceNTD: "1500-1800",
      targetUsers: ["商務人士", "科技玩家", "多裝置出差族"],
    };

    //-----------------------
    // ⑥ Prompt（包含三家廠商）
    //-----------------------
    const futureIndexPrompt = `
你是一位「行動電源產業」的策略研究 AI 顧問。

現在請你根據下列資料，評估三個品牌（我方新品 Fusion-20、Anker、Belkin）的下列五個維度：

1. 技術前瞻性 (technical_futurity)
2. 市場契合度 (market_fit)
3. 成本效益 (cost_effectiveness)
4. 設計美學 (design_aesthetics)
5. 競爭壁壘 (competitive_moat)

所有分數請用 0~100 分，並提供一句中文理由。

=== 我方新品（假定） ===
${JSON.stringify(fusion20Spec, null, 2)}

=== 技術趨勢資料（未來 6 個月預測） ===
${JSON.stringify(techData, null, 2)}

=== 使用者需求趨勢與熱力圖資料 ===
${JSON.stringify(userRequirementsData, null, 2)}

=== NLP 分析（整體 ABSA） ===
${JSON.stringify(absaAll, null, 2)}

=== NLP（PChome 價格帶 ABSA） ===
${JSON.stringify(pchomeAbsaData, null, 2)}

=== NLP（情緒分佈 sentiment_distribution） ===
${JSON.stringify(pchomeSentimentDistData, null, 2)}

=== 價格帶爬蟲評論（成本效益依據） ===
${JSON.stringify(reviewData, null, 2)}

請產出 JSON（嚴格遵守結構）：

{
  "product": {
    "name": "Fusion-20",
    "brand": "PowerPulse",
    "scores": {
      "technical_futurity": { "score": 0, "reason": "" },
      "market_fit": { "score": 0, "reason": "" },
      "cost_effectiveness": { "score": 0, "reason": "" },
      "design_aesthetics": { "score": 0, "reason": "" },
      "competitive_moat": { "score": 0, "reason": "" }
    }
  },
  "competitors": [
    {
      "name": "Anker 高階 GaN 行動電源",
      "brand": "Anker",
      "scores": {}
    },
    {
      "name": "Belkin 高階 PD / MagSafe 行動電源",
      "brand": "Belkin",
      "scores": {}
    }
  ]
}
    `;

    //-----------------------
    // ⑦ Gemini 呼叫
    //-----------------------
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
    });

    const result = await model.generateContent(futureIndexPrompt);
    const rawText = result?.response?.text() ?? "";

    //-----------------------
    // ⑧ JSON parsing
    //-----------------------
    let parsed = null;
    const jsonString = extractJson(rawText);

    if (jsonString) {
      try {
        parsed = JSON.parse(jsonString);
      } catch (err) {
        console.error("JSON parse error:", err);
      }
    }

    //-----------------------
    // ⑨ 整理我方五指標給前端
    //-----------------------
    let futureIndex = null;
    if (parsed?.product?.scores) {
      const s = parsed.product.scores;
      futureIndex = [
        { label: "技術前瞻性", key: "technical_futurity", score: s.technical_futurity?.score },
        { label: "市場契合度", key: "market_fit", score: s.market_fit?.score },
        { label: "成本效益", key: "cost_effectiveness", score: s.cost_effectiveness?.score },
        { label: "設計美學", key: "design_aesthetics", score: s.design_aesthetics?.score },
        { label: "競爭壁壘", key: "competitive_moat", score: s.competitive_moat?.score },
      ];
    }

    res.json({
      ok: true,
      prompt: futureIndexPrompt,
      rawText,
      parsed,
      futureIndex,
    });

  } catch (err) {
    console.error("FUTURE INDEX ERROR:", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

//---------------------------------------
// 啟動伺服器
//---------------------------------------
app.listen(PORT, () => {
  console.log(`Future Index API running at http://localhost:${PORT}/api/future-index`);
});
