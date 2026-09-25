/* Illustrative arithmetic only. No experiment files or network requests are used. */
(function () {
  'use strict';
  function contrasts(delta, cleanGain, degradedGain, observationShift) {
    const reference = [2, 2, 2 + delta, 2 + delta]; // A00, A01, A10, A11
    const estimate = [2, 2 + observationShift, 2 + cleanGain * delta,
      2 + degradedGain * delta + observationShift];
    const error = estimate.map((x, i) => x - reference[i]);
    return { reference, estimate,
      cleanResponse: error[2] - error[0], degradedResponse: error[3] - error[1],
      observationA: error[1] - error[0], observationB: error[3] - error[2],
      interaction: error[3] - error[1] - error[2] + error[0] };
  }
  function coupling(u, v) {
    return { delta: (v-u)*(v-u)/2, endpoint: (u*u+v*v)/2, cross: -u*v };
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = { contrasts, coupling };
  if (typeof document === 'undefined') return;
  const ids = ['true-change', 'clean-gain', 'degraded-gain', 'observation-shift'];
  const signed = x => (x > 0 ? '+' : '') + (Math.abs(x) < 0.00001 ? 0 : x).toFixed(2) + '°';
  function update() {
    const values = ids.map(id => Number(document.getElementById(id).value));
    const c = contrasts(values[0], values[1] / 100, values[2] / 100, values[3]);
    ids.forEach((id, i) => {
      document.getElementById(id + '-value').textContent = i === 1 || i === 2 ? values[i] + '%' : signed(values[i]);
    });
    c.reference.forEach((v, i) => {
      document.getElementById('reference-' + i).textContent = signed(v);
      document.getElementById('estimate-' + i).textContent = signed(c.estimate[i]);
    });
    [['clean-response', c.cleanResponse], ['degraded-response', c.degradedResponse],
     ['observation-a', c.observationA], ['observation-b', c.observationB],
     ['interaction', c.interaction]].forEach(([id, value]) => {
      document.getElementById(id).textContent = signed(value);
    });
    document.getElementById('clean-absolute').textContent = Math.abs(c.cleanResponse).toFixed(2) + '°';
    document.getElementById('degraded-absolute').textContent = Math.abs(c.degradedResponse).toFixed(2) + '°';
    document.getElementById('explorer-description').textContent =
      'The reference change is ' + signed(values[0]) + '. The estimated change is ' +
      signed(values[0] * values[1] / 100) + ' under clean observation and ' +
      signed(values[0] * values[2] / 100) + ' under degraded observation. The difference between their signed response biases is ' +
      signed(c.interaction) + '.';
  }
  ids.forEach(id => document.getElementById(id).addEventListener('input', update));
  const presets = { faithful: [4, 100, 100, 0], attenuated: [4, 50, 50, 0], sensitive: [4, 100, 50, 1] };
  document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
    presets[button.dataset.preset].forEach((v, i) => { document.getElementById(ids[i]).value = v; }); update();
  }));
  update();
  function updateCoupling() {
    const u = Number(document.getElementById('residual-a').value);
    const v = Number(document.getElementById('residual-b').value);
    const c = coupling(u, v);
    [['a', u], ['b', v]].forEach(([endpoint, value]) => {
      document.getElementById('residual-' + endpoint + '-value').textContent = value.toFixed(2);
      [['x', value], ['y', -value]].forEach(([channel, residual]) => {
        const bar = document.getElementById('residual-' + endpoint + '-' + channel);
        bar.setAttribute('x', 300 + Math.min(0, residual) * 40);
        bar.setAttribute('width', Math.abs(residual) * 40);
      });
    });
    document.getElementById('delta-loss').textContent = c.delta.toFixed(2);
    document.getElementById('endpoint-loss').textContent = c.endpoint.toFixed(2);
    document.getElementById('coupling-term').textContent = c.cross.toFixed(2);
    document.getElementById('coupling-description').textContent =
      c.delta === 0 && c.endpoint > 0 ? 'Both states have the same nonzero error. Coupling permits this shared bias, so its loss is zero while endpoint regression still penalizes it.' :
      c.endpoint === 0 ? 'Both endpoint errors are zero. Both auxiliary losses are zero.' :
      'The error difference contributes ' + c.delta.toFixed(2) + ' to the coupled loss; the independent endpoint errors contribute ' + c.endpoint.toFixed(2) + '. These are illustrative feature units.';
  }
  ['residual-a', 'residual-b'].forEach(id => document.getElementById(id).addEventListener('input', updateCoupling));
  const couplingPresets = { shared: [2, 2], opposed: [2, -2], zero: [0, 0] };
  document.querySelectorAll('[data-coupling-preset]').forEach(button => button.addEventListener('click', () => {
    const values = couplingPresets[button.dataset.couplingPreset];
    document.getElementById('residual-a').value = values[0];
    document.getElementById('residual-b').value = values[1];
    updateCoupling();
  }));
  updateCoupling();
  document.querySelectorAll('.explorer input, .explorer button').forEach(control => { control.disabled = false; });
  document.querySelectorAll('.mobile-nav a').forEach(link => link.addEventListener('click', () => {
    document.querySelector('.mobile-nav').open = false;
  }));

  const modal = document.getElementById('figure-dialog');
  if (modal && typeof modal.showModal === 'function') {
    document.querySelectorAll('a.figure-link').forEach(link => link.addEventListener('click', event => {
      event.preventDefault();
      document.getElementById('large-figure').src = link.href;
      document.getElementById('large-figure').alt = link.querySelector('img').alt;
      document.getElementById('figure-description').textContent = link.querySelector('img').alt;
      modal.showModal();
    }));
    document.getElementById('close-figure').addEventListener('click', () => modal.close());
    modal.addEventListener('click', e => { if (e.target === modal) modal.close(); });
  }
  if (typeof IntersectionObserver !== 'undefined') {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) {
        document.querySelectorAll('.toc a').forEach(a => {
          if (a.hash === '#' + entry.target.id) a.setAttribute('aria-current', 'location');
          else a.removeAttribute('aria-current');
        });
      }
    }, { rootMargin: '0px 0px -65% 0px' });
    document.querySelectorAll('main h2[id]').forEach(h => observer.observe(h));
  }
})();
