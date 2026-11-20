// Every lesson remains readable when scripting is disabled.
(() => {
  const lessons = [...document.querySelectorAll('.lesson')];
  const menuLinks = [...document.querySelectorAll('.menu a')];
  const quiz = {
    start: ['One video supplies eight windows. How many source recordings does it contribute?', ['One', 'Eight', 'We need to know the frame rate.'], 0, ['Yes. Cutting a video into windows leaves its source identity unchanged.', 'Eight is the number of windows. They share a source.', 'The frame rate tells you when observations occurred. The recording still has one source.']],
    classifier: ['What supplies the target when the student tries to predict a deliberately hidden joint?', ['A medical diagnosis', 'The teacher’s features for the observed joint', 'A guess for a joint the camera never saw'], 1, ['A diagnosis is unnecessary for this training target. Look for a measurement already in the clip.', 'Yes. The teacher sees the observation that was hidden from the student.', 'A camera dropout supplies no measurement. The experiment hides joints it did observe.']],
    restart: ['A model forecasts an ankle more accurately than copying its last position. Which question still needs a separate test?', ['Which forecast had smaller error?', 'Was the ankle visible in the example?', 'Can forecast errors distinguish gait labels on new recordings?'], 2, ['The comparison already answers which forecast had smaller error.', 'Visibility is a measurement check. Consider the separate task that uses gait labels.', 'Yes. Forecasting coordinates and ranking gait labels have different targets and different controls.']],
    forecast: ['Why compare an anomaly score with a score based only on how little the joints moved?', ['To see whether the model adds information beyond movement amount', 'To count windows as new participants', 'To remove all slow walkers from the study'], 0, ['Yes. A score can track a simple movement difference even when it reveals little about coordination.', 'Both scores must be compared on the same independent recordings or participants.', 'The comparison keeps the observations and changes how they are scored.']]
  };

  function showLesson(focus = false) {
    const id = location.hash.slice(1) || 'start';
    const target = document.getElementById(id);
    const current = target?.closest('.lesson') ||
      (id === 'main' && lessons.find(page => !page.hidden)) || lessons[0];
    lessons.forEach(page => { page.hidden = page !== current; });
    menuLinks.forEach(link => {
      if (link.hash === '#' + current.id) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    document.title = current.querySelector('h1').textContent + ' · AlexPose';
    if (focus) {
      const heading = target && current.contains(target) && target !== current ? target : current.querySelector('h1');
      heading.setAttribute('tabindex','-1');
      heading.focus({preventScroll:true});
      heading.scrollIntoView({block:'start'});
    }
  }

  lessons.forEach((page, index) => {
    const headings = [...page.querySelectorAll('h2,h3')].filter(el => /^\d{2} ·/.test(el.textContent));
    if (headings.length) {
      const box = page.querySelector('.jump');
      const label = document.createElement('label');
      label.textContent = 'Jump to a notebook';
      label.htmlFor = 'jump-' + page.id;
      const select = document.createElement('select');
      select.id = label.htmlFor;
      select.add(new Option('Choose a notebook…',''));
      headings.forEach(heading => {
        heading.id = page.id + '-' + heading.textContent.slice(0,2);
        select.add(new Option(heading.textContent, heading.id));
      });
      select.addEventListener('change', () => {
        if (select.value) {
          location.hash = select.value;
          select.selectedIndex = 0;
        }
      });
      box.append(label, select);
    }

    const [prompt, choices, correct, reasoning] = quiz[page.id];
    const area = page.querySelector('.quiz');
    area.innerHTML = '<h2>Check your understanding</h2><p></p><div class="choices"></div><div class="readout" role="status" hidden></div>';
    area.querySelector('p').textContent = prompt;
    const buttons = area.querySelector('.choices');
    const feedback = area.querySelector('.readout');
    choices.forEach((choice, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = choice;
      button.setAttribute('aria-pressed','false');
      button.addEventListener('click', () => {
        [...buttons.children].forEach(el => el.setAttribute('aria-pressed', String(el === button)));
        feedback.hidden = false;
        feedback.textContent = (i === correct ? '' : 'Try again. ') + reasoning[i];
      });
      buttons.append(button);
    });
    const footer = page.querySelector('.footnav');
    const neighbor = (idx, label, primary) => {
      const link = document.createElement('a');
      link.className = 'next' + (primary ? ' primary' : '');
      link.href = '#' + lessons[idx].id;
      link.textContent = label;
      footer.append(link);
    };
    if (index) neighbor(index-1, '← Previous lesson',false);
    else footer.append(document.createElement('span'));
    if (index+1 < lessons.length) neighbor(index+1, 'Next lesson →', true);
    else neighbor(0,'Return to the guide ↑',false);
  });
  showLesson();
  window.addEventListener('hashchange', () => showLesson(true));
})();
