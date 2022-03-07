// optim.js
// by Yusuke Shinyama
// License: CC-SA-4.0

// x = [5, 0, 10, 7, 10, 2, 4, 8, 0, 9];

// fb
function fb() {
  let x = [];
  for (let i = 0; i <= 9; i++) {
    let e = document.getElementById("b"+i);
    x.push(parseInt(e.value));
  }
  let y = (
    (x[3]-x[0]-2)**2 + (x[1])**2 + (x[2]-x[4])**2 + (x[6]-2*x[5])**2 + (x[7]-2*x[6])**2 +
    (x[1]-x[8])**2 + (x[9]-9)**2 + (x[1]+x[2]-10)**2 + (x[7]-x[0]-3)**2 + (x[9]-x[7]-1)**2
  );
  console.log("fb("+x+") = "+y);
  let bout = document.getElementById("bout");
  bout.innerHTML = y.toString();
}

// fc
function fc() {
  let x = [];
  for (let i = 0; i <= 9; i++) {
    let e = document.getElementById("c"+i);
    x.push(parseInt(e.value));
  }
  let y = (
    (x[3]-x[0]-2)**2 + (x[1])**2 + (x[2]-x[4])**2 + (x[6]-2*x[5])**2 + (x[7]-2*x[6])**2 +
    (x[1]-x[8])**2 + (x[9]-9)**2 + (x[1]+x[2]-10)**2 + (x[7]-x[0]-3)**2 + (x[9]-x[7]-1)**2
  );
  let dy = [
    2*x[0]-x[3]-x[7]+5, 3*x[1]+x[2]-x[8]-10, 2*x[2]+x[1]-x[4]-10, x[3]-x[0]-2, -x[2]+x[4],
    -2*(x[6]-2*x[5]), 5*x[6]-2*x[5]-2*x[7], 3*x[7]-2*x[6]-x[0]-x[9]-2, -x[1]+x[8], 2*x[9]-x[7]-10
  ];
  console.log("fc("+x+") = "+y, dy);
  let cout = document.getElementById("cout");
  cout.innerHTML = y.toString();
  for (let i = 0; i <= 9; i++) {
    let e = document.getElementById("d"+i);
    let v = dy[i];
    if (0 < v) {
      e.innerHTML = "↓";
    } else if (v < 0) {
      e.innerHTML = "↑";
    } else {
      e.innerHTML = "&nbsp;";
    }
  }
}

function main() {
  fb();
  fc();
}
