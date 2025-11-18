import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";
import path from "path";
import { readFileSync } from "fs";
import Papa from "papaparse";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();


const app = express();
const PORT = 8000;

// 讓 __dirname 在 ES module 中可用
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 🔥 使用相對路徑：往上一層找到 AICompetition
const ROOT = path.resolve(__dirname, "..");

function readCSV(filePath) {
  const content = readFileSync(filePath, "utf8");
  return Papa.parse(content, { header: true }).data;
}

// ---------- AI Insight API ----------
app.get("/api/insight", async (req, res) => {
  try {
    // ① 技術趨勢
    const techFiles = ["GaN.csv", "magsafe.csv", "pd.csv", "typeC.csv"];
    const techData = techFiles.map(file => ({
      category: file.replace(".csv", ""),
      rows: readCSV(`${ROOT}/timeseries_TechTrends/${file}`)
    }));

    // ② NLP / ABSA
    const absaData = readCSV(`${ROOT}/nlp_result/absa_results.csv`);

    // ③ 價格帶評論 CSV
    const reviewFiles = [
      "500.csv",
      "1000.csv",
      "1500.csv",
      "2000.csv",
      "2000_10000.csv",
      "10000.csv",
    ];

    const reviewData = reviewFiles.map(file => ({
      priceRange: file.replace(".csv", ""),
      rows: readCSV(`${ROOT}/crawlers_result/${file}`)
    }));

    // ---------- Prompt ----------
    const prompt = `
你是一位行動電源產業分析 AI，請根據以下資料生成「AI 策略洞察摘要」，限制 180 字，語氣自然。

=== 技術趨勢資料（GaN / PD / MagSafe / TypeC） ===
${JSON.stringify(techData, null, 2)}

=== NLP ABSA（前 40 筆樣本） ===
${JSON.stringify(absaData.slice(0, 40), null, 2)}

=== 價格帶評論（依價格帶） ===
價格邏輯：
- 500：500 以下
- 1000：500–1000
- 1500：1000–1500
- 2000：1500–2000
- 2000_10000：2000–10000
- 10000：10000 以上
${JSON.stringify(reviewData, null, 2)}

請輸出單一段落，包含：
- 技術聲量變化（GaN、PD、MagSafe、TypeC）
- ABSA 的痛點/亮點（重量、發熱、容量、材質、充電線）
- 不同價格帶消費者的行為差異
- 最後總結市場策略洞察（限 180 字）。
`;

    // ---------- Gemini 呼叫 ----------
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    const result = await model.generateContent(prompt);
    const insight = result?.response?.text() ?? "AI 無法產生摘要";

    res.json({ insight });

  } catch (err) {
    console.error("INSIGHT ERROR:", err);
    res.status(500).json({ error: err.message });
  }
});

// ---------- 啟動 API ----------
app.listen(PORT, () => {
  console.log(`AI Insight API running at http://localhost:${PORT}/api/insight`);
});
