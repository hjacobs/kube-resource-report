// https://bulma.io/documentation/components/navbar/
document.addEventListener('DOMContentLoaded', () => {

  // Get all "navbar-burger" elements
  const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);

  // Check if there are any navbar burgers
  if ($navbarBurgers.length > 0) {

    // Add a click event on each of them
    $navbarBurgers.forEach( el => {
      el.addEventListener('click', () => {

        // Get the target from the "data-target" attribute
        const target = el.dataset.target;
        const $target = document.getElementById(target);

        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        el.classList.toggle('is-active');
        $target.classList.toggle('is-active');

      });
    });
  }

  const $collapsibleHeaders = Array.prototype.slice.call(document.querySelectorAll('main .collapsible h2.title'), 0);

  $collapsibleHeaders.forEach( el => {
    el.addEventListener('click', function () {
      const $section = el.parentElement;
      $section.classList.toggle('is-collapsed');
      const $collapsed = Array.prototype.slice.call(document.querySelectorAll('main .is-collapsed'), 0);
      const names = [];
      $collapsed.forEach( el => {
          names.push(el.dataset.name);
      });
      if (names) {
        document.location.hash = "collapsed=" + names.join(",");
      } else {
        document.location.hash = "";
      }
    });
  });

  const hash = document.location.hash;
  if (hash) {
    const hashParams = hash.substring(1).split(";");
    hashParams.forEach( param => {
        const keyVal = param.split("=");
        if (keyVal[0] == "collapsed") {
            // collapse all sections mentioned in URL fragment
            keyVal[1].split(",").forEach( name => {
                const $sections = document.querySelectorAll('main .collapsible[data-name=' + name + ']');
                $sections.forEach( el => {
                    el.classList.add("is-collapsed");
                });
            });
        }
    });
  }

});
