// optim.js
// by Yusuke Shinyama
// License: CC-SA-4.0

// x = [5, 0, 10, 7, 10, 2, 4, 8, 0, 9];

class Optim {

  sliders1 = [];
  sliders2 = [];
  result1 = null;
  result2 = null;
  grads2 = [];

  setup(id1, id2) {
    let obj = this;
    let parent = document.getElementById(id1);
    this.result1 = parent.getElementsByClassName('result')[0];
    for (let e of parent.getElementsByTagName('input')) {
      this.sliders1.push(e)
      e.addEventListener('input', () => { obj.update1(); }, false);
    }

    parent = document.getElementById(id2);
    this.result2 = parent.getElementsByClassName('result')[0];
    for (let e of parent.getElementsByTagName('input')) {
      this.sliders2.push(e)
      e.addEventListener('input', () => { obj.update2(); }, false);
    }
    for (let e of parent.getElementsByClassName('grad')) {
      this.grads2.push(e);
    }

    this.update1();
    this.update2();
  }

  update1() {
    let x = [];
    for (let e of this.sliders1) {
      x.push(parseInt(e.value));
    }
    let y = (
      (x[3]-x[0]-2)**2 + (x[1])**2 + (x[2]-x[4])**2 + (x[6]-2*x[5])**2 + (x[7]-2*x[6])**2 +
        (x[1]-x[8])**2 + (x[9]-9)**2 + (x[1]+x[2]-10)**2 + (x[7]-x[0]-3)**2 + (x[9]-x[7]-1)**2
    );
    console.log("f1("+x+") = "+y);
    this.result1.innerHTML = y.toString();
  }

  update2() {
    let x = [];
    for (let e of this.sliders2) {
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
    console.log("f2("+x+") = "+y, dy);
    this.result2.innerHTML = y.toString();
    for (let i = 0; i < this.grads2.length; i++) {
      let e = this.grads2[i];
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
}

var optim = new Optim();
function optim_setup(id1, id2) {
  optim.setup(id1, id2);
}
