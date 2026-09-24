#target photoshop

/**
 * 计算当前选区面积占整张图片面积的百分比
 * 支持任意形状选区（包括不规则、羽化边缘等）
 * 通过临时Alpha通道统计像素，对选区边缘按亮度阈值（≥128）判定为选中区域
 */

function calculateSelectionAreaRatio() {
    // 获取当前活动文档
    var doc = app.activeDocument;
    if (doc == null) {
        alert("未检测到打开的文档，请先打开一张图片。");
        return;
    }

    // 检查是否存在选区
    var sel = doc.selection;
    try {
        var bounds = sel.bounds;
        if (bounds == null || bounds.length === 0) {
            alert("当前文档中没有选区，请先使用选区工具创建一个选区。");
            return;
        }
    } catch (e) {
        alert("无法获取选区信息，请确保存在有效选区。");
        return;
    }

    // 获取文档总像素数（转换为像素单位，避免标尺单位干扰）
    var widthPx = doc.width.as("px");
    var heightPx = doc.height.as("px");
    var totalPixels = widthPx * heightPx;

    // 保存当前历史状态，用于脚本结束后恢复（不留痕迹）
    var originalHistory = doc.activeHistoryState;

    // 添加临时Alpha通道
    var tempChannel = doc.channels.add();
    tempChannel.name = "TempSelectionArea";

    // 设置前景色为白色（用于填充选区）
    var originalForeground = app.foregroundColor;
    var whiteColor = new SolidColor();
    whiteColor.rgb.red = 255;
    whiteColor.rgb.green = 255;
    whiteColor.rgb.blue = 255;
    app.foregroundColor = whiteColor;

    // 激活临时通道，并将当前选区填充为白色（选中区域）
    doc.activeChannels = [tempChannel];
    doc.selection.fill(app.foregroundColor);

    // 恢复原前景色
    app.foregroundColor = originalForeground;

    // 获取临时通道的直方图（0–255各亮度级像素数量）
    var hist = tempChannel.histogram;

    // 统计亮度≥128的像素数（视为被选区覆盖的有效像素）
    var selectedPixels = 0;
    for (var i = 128; i <= 255; i++) {
        selectedPixels += hist[i];
    }

    // 恢复到脚本运行前的历史状态（自动删除临时通道，撤销所有改动）
    doc.activeHistoryState = originalHistory;

    // 计算百分比
    var ratio = (selectedPixels / totalPixels) * 100;
    var ratioFixed = ratio.toFixed(2);

    // 显示结果
    var message = "📐 选区面积占比: " + ratioFixed + "%\n" +
                  "🖌️ 选区覆盖像素数: " + selectedPixels.toLocaleString() + "\n" +
                  "🖼️ 文档总像素数: " + totalPixels.toLocaleString();
    alert(message, "选区面积占比统计");
}

// 执行主函数
calculateSelectionAreaRatio();