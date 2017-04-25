var express = require('express')
var expressMongoRest = require('express-mongo-rest')
var app = express()

app.use('/api', expressMongoRest('mongodb://localhost:27017/handwriting-db'))

/* serves main page */
app.get("/", function(req, res) {
    res.sendfile('index.html')
});

/* serves all the static files */
app.get(/^(.+)$/, function(req, res){ 
    console.log('static file request : ' + req.params);
    res.sendfile( __dirname + req.params[0]); 
});

var server = app.listen(3000, function () {
    console.log('Listening on Port', server.address().port)
})