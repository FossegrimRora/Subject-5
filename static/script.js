document.querySelector("form").addEventListener("submit",function(e){

let inputs=document.querySelectorAll("input");

for(let input of inputs){

if(input.value===""){

alert("Semua data harus diisi.");

e.preventDefault();

return;

}

}

});