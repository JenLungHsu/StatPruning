# 安裝 R2sample（如果尚未安裝）
install.packages("R2sample")

library(R2sample)

# === STEP 1: 預先處理資料（去除離群值）===
remove_outliers <- function(v) {
  q1 <- quantile(v, 0.25)
  q3 <- quantile(v, 0.75)
  iqr <- q3 - q1
  v[v >= (q1 - 1.5 * iqr) & v <= (q3 + 1.5 * iqr)]
}

x_clean <- remove_outliers(x_raw)
y_clean <- remove_outliers(y_raw)

# === STEP 2: 定義重抽樣資料產生器函數 ===
# dummy 參數只是占位用，twosample_power 會自動傳入
f <- function(dummy) {
  list(
    x = sample(x_clean, 300, replace = FALSE),
    y = sample(y_clean, 300, replace = FALSE)
  )
}

# === STEP 3: 執行 power 模擬 ===
# 這裡 dummy=1:200 表示模擬 200 次，B 是置換檢定的次數
pwr_result <- twosample_power(
  f = f,
  dummy = 1:200,
  B = 500,              # 每次置換次數（愈高愈準確但較慢）
  alpha = 0.05,
  doMethods = c("KS", "AD", "CvM", "LR", "ZA", "ZK", "ZC", "Wassp1")
)

# === STEP 4: 繪製 power 結果圖表 ===
plot_power(pwr_result)

# 若你想看平均 power：
colMeans(pwr_result, na.rm = TRUE)
