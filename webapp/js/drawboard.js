$(function() {
    var drawboardCanvas = document.getElementById("drawboardCanvas");
    var curColor = $('#selectColor option:selected').val();
    if(drawboardCanvas) {
        var isDown = false;
        var ctx = drawboardCanvas.getContext("2d");
        var canvasX, canvasY;
        ctx.lineWidth = 5;
            
        $(drawboardCanvas)
            .bind( "touchstart mousedown", function(e){
                isDown = true;
                ctx.beginPath();
                canvasX = e.pageX - drawboardCanvas.offsetLeft;
                canvasY = e.pageY - drawboardCanvas.offsetTop;
                ctx.moveTo(canvasX, canvasY);
            });
        $(drawboardCanvas)
            .bind( "touchend mouseup touchleave mouseleave", function(e){
                isDown = false;
                ctx.closePath();
            });
        $(drawboardCanvas)
            .bind( "touchmove mousemove", function(e){
                if(isDown != false) {
                    canvasX = e.pageX - drawboardCanvas.offsetLeft;
                    canvasY = e.pageY - drawboardCanvas.offsetTop;
                    ctx.lineTo(canvasX, canvasY);
                    ctx.strokeStyle = curColor;
                    ctx.stroke();
                }
            });
    }
        
    $('#selectColor').change(function () {
        curColor = $('#selectColor option:selected').val();
    });
});