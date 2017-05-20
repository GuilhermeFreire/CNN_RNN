const http = require("http");
const fs = require("fs");
const port = 80;

let db = fs.readFileSync("./database.json");//TODO: try catch
db = JSON.parse(db);

const requestHandler = (request, response) => {
    let finalResponse = null;

    console.log(request.url);

    if(request.url == "/"){
        finalResponse = fs.readFileSync('./index.html');//TODO: try catch
    }else if(request.url == "/sendResults") {
        let body = [];
        request.on('data', function(chunk) {
            body.push(chunk);
        }).on('end', function() {
            body = Buffer.concat(body).toString();

            console.log(body);
            let data = JSON.parse(body);
            data.image = data.image.split("/");
            data.image = data.image[data.image.length - 1];
            if(db[data.image] == undefined){
                db[data.image] = {
                    "123": 0,
                    "132": 0,
                    "213": 0,
                    "231": 0,
                    "312": 0,
                    "321": 0
                }
            }
            let field = ""+data.first+data.second+data.third;
            db[data.image][field] += 1;

            fs.writeFile("./database.json", JSON.stringify(db), function (err) {
                if(err){
                    response.end(JSON.stringify({success: false}));
                }else{
                    response.end(JSON.stringify({success: true}));
                }
            });
        });
        return;
    }else{
        if(fs.existsSync("."+request.url)){//TODO: try catch
            finalResponse = fs.readFileSync("."+request.url);//TODO: try catch
        }else{
            response.writeHeader(404);
        }
    }

    response.end(finalResponse);
};
// Content-Disposition: attachment; filename="fname.ext"
const server = http.createServer(requestHandler);

server.listen(port, (err) => {
    if(err){
        return console.log("Something bad happened", err)
    }
    console.log(`Server is listening on ${port}`)
});