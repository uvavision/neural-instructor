// The Common module is designed as an auxiliary module
// to hold functions that are used in multiple other modules
/* eslint no-unused-vars: "off" */

var Common = (function() {
  // Publicly accessible methods defined
  return {drawCanvasData: drawCanvasData};

    function drawCanvasData(ctx, canvas_data, scale) {
        // ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
        // ctx.lineWidth = 3;
        canvas_data.forEach(function (box_anno) {
            bbox = [box_anno['left']*scale, box_anno['top']*scale, box_anno['width']*scale, box_anno['height']*scale];
            color = box_anno['label'];
            shape = box_anno['shape'];
            ctx.fillStyle = color;
            if (shape === "circle") {
              var centerX = bbox[0] + bbox[2]*0.5;
              var centerY = bbox[1] + bbox[3]*0.5;
              var radius = bbox[2] * 0.5;
              ctx.beginPath();
              ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
              // ctx.fillStyle = color;
              ctx.fill();
            } else if (shape === "square") {
              ctx.fillRect(bbox[0], bbox[1], (bbox[2]), (bbox[3]));
            } else if (shape === "triangle") {
              ctx.beginPath();
              ctx.moveTo(bbox[0]+bbox[2]*0.5, bbox[1]);
              ctx.lineTo(bbox[0]+bbox[2], bbox[1]+bbox[3]);
              ctx.lineTo(bbox[0], bbox[1]+bbox[3]);
              ctx.fill();
            } else {
              alert("unknown shape");
            }
            // ctx.rect(bbox[0] * scale, bbox[1] * scale, (bbox[2]) * scale, (bbox[3]) * scale);
            // ctx.stroke();
            // drawTextBG(ctx, box_anno['label'], '15px arial', bbox[0] * scale, bbox[1] * scale);
        });
    }
}());
