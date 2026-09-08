const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = '/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project';
const FIG = path.join(ROOT, 'figures');
const TFIG = path.join(ROOT, 'tcca', 'figures');
const CFIG = path.join(ROOT, 'cca', 'figures');
const RFIG = path.join(ROOT, 'rl_model', 'figures');
const LFIG = path.join(ROOT, 'lfp', 'figures');
const OUT = path.join(ROOT, 'presentations', 'StriatumUpdate_20260908.pptx');

// The four RS panels used to be excluded here because they rendered empty.
// Fixed 2026-08-11/12 (probe-2 waveforms now loaded, cortical/hippocampal
// units classified FS vs RS, and the plotter no longer collapses code 5),
// so they carry real data and are included like everything else.
const EXCLUDE = new Set();

// PNG pixel dimensions straight from the IHDR chunk (bytes 16-24).
function pngSize(fp) {
  const fd = fs.openSync(fp, 'r');
  const buf = Buffer.alloc(24);
  fs.readSync(fd, buf, 0, 24, 0);
  fs.closeSync(fd);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

const num = f => { const m = f.match(/_(\d+)_/); return m ? parseInt(m[1], 10) : 1e9; };
const byNum = (a, b) => num(a) - num(b) || a.localeCompare(b);

// A sweep that reran after figures gained Name properties leaves BOTH the old
// "<prefix>_NN_fig.png" and the new "<prefix>_NN_<name>.png" on disk — same
// slot, different vintage, and for ensemble_58 the older one is the broken
// pre-fix KO panel. Keep only the newest file per (prefix, slot number).
function dedupeBySlot(files) {
  const best = new Map();
  const loose = [];
  for (const f of files) {
    const m = f.match(/^([a-z0-9_]+?)_(\d+)_/);
    if (!m) { loose.push(f); continue; }
    const key = `${m[1]}#${m[2]}`;
    const mt = fs.statSync(path.join(FIG, f)).mtimeMs;
    const cur = best.get(key);
    if (!cur || mt > cur.mt) best.set(key, { f, mt });
  }
  return [...[...best.values()].map(v => v.f), ...loose];
}

const all = dedupeBySlot(
  fs.readdirSync(FIG).filter(f => f.endsWith('.png') && !EXCLUDE.has(f))
);

const pick = (re, sort = byNum) => all.filter(f => re.test(f)).sort(sort);

// Same as `pick`, for a directory other than FIG. `order` optionally pins the
// leading files; anything else matching falls in behind, alphabetically.
function pickIn(dir, re, order = []) {
  const match = typeof re === 'function' ? re : f => re.test(f);
  const hits = fs.readdirSync(dir).filter(f => f.endsWith('.png') && match(f));
  const rank = f => { const i = order.indexOf(f); return i < 0 ? order.length : i; };
  return hits.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
             .map(f => path.join(dir, f));
}

// LFP figures predating the 2026-08-28 cohort split have no `_task`/`_control`
// suffix and are superseded by the suffixed pair; the July decoding/CCA pair is
// the provisional 4-animal version superseded by the full-cohort run. An "all
// results" deck must not carry either. (2026-09-08)
const LFP_SUPERSEDED = f => {
  if (/^(decode_position|cca_cross_area)\.png$/.test(f)) return true;
  const stem = f.replace(/\.png$/, '');
  if (/_(task|control)$/.test(stem) || /_task_vs_control$/.test(stem)) return false;
  return fs.existsSync(path.join(LFIG, `${stem}_task.png`)) ||
         fs.existsSync(path.join(LFIG, `${stem}_control.png`));
};
const range = (pre, lo, hi) =>
  all.filter(f => f.startsWith(pre) && num(f) >= lo && num(f) <= hi).sort(byNum);

const SECTIONS = [
  { title: 'Cohort summary: animals, units, areas and cell types',
    files: pick(/^summary_\d+_/).map(f => path.join(FIG, f)) },

  { title: 'Preprocessing: choosing the spatial bin size',
    files: pick(/^compare_bin_sizes\.png$/).map(f => path.join(FIG, f)) },

  { title: 'Behaviour and learning trajectories',
    files: range('integrated_', 1, 8).map(f => path.join(FIG, f)) },

  { title: 'Trial-to-trial reliability of neural responses',
    files: [...range('integrated_', 9, 10), ...range('integrated_', 23, 25)].map(f => path.join(FIG, f)) },

  { title: 'Position decoding and its evolution across learning',
    files: range('integrated_', 11, 14).map(f => path.join(FIG, f)) },

  { title: 'Spatial tuning by area',
    files: range('integrated_', 15, 22).map(f => path.join(FIG, f)) },

  { title: 'Spatiotemporal activity: task animals',
    files: pick(/^spatiotemporal_\d+_task_/).map(f => path.join(FIG, f)) },

  { title: 'Spatiotemporal activity: control animals',
    files: pick(/^spatiotemporal_\d+_control_(?!2_)/).map(f => path.join(FIG, f)) },

  { title: 'Control 2 (dark-only cohort): dark-period profile and evolution',
    files: pick(/^spatiotemporal_\d+_control_2_/).map(f => path.join(FIG, f)) },

  { title: 'Corridor versus dark: is the dark ITI a baseline?',
    files: pickIn(FIG, f => /^CorridorVsDark_/.test(f) ||
                    /^corridordark_\d+_(?!corridordark_)/.test(f),
                  ['CorridorVsDark_velocity.png',
                   'CorridorVsDark_zscore_variants.png',
                   'CorridorVsDark_zscore_variants_control1.png',
                   'CorridorVsDark_RawFR_Hierarchical_byArea.png',
                   'CorridorVsDark_ZScored_Hierarchical_byArea.png']) },

  { title: 'TCA: rank selection and components',
    files: ['tca_bic_diagnostics.png', 'tca_components_rank4.png', 'tca_components_rank5.png',
            'tca_unbalanced_components_rank4.png', 'tca_unbalanced_components_rank5.png',
            'tca_balance_comparison.png']
           .filter(f => all.includes(f)).map(f => path.join(FIG, f)) },

  { title: 'TCA: task versus control decompositions',
    files: ['tca_taskonly_components_rank5.png', 'tca_taskonly_components_rank4.png',
            'tca_controlonly_components_rank2.png', 'tca_controlonly_components_rank4.png',
            'tca_task_vs_control_trialfactors.png']
           .filter(f => all.includes(f)).map(f => path.join(FIG, f)) },

  { title: 'TCA pipeline outputs: combined tensor',
    files: pick(/^tca_\d+_/).map(f => path.join(FIG, f)) },

  { title: 'TCA pipeline outputs: task-only and control-only tensors',
    files: pick(/^tca_(task|control)_unbalanced_\d+_/).map(f => path.join(FIG, f)) },

  { title: 'Ensembles: composition, purity and coupling',
    files: range('ensemble_', 1, 12).map(f => path.join(FIG, f)) },

  { title: 'Ensembles: spatial and temporal profiles',
    files: range('ensemble_', 13, 52).map(f => path.join(FIG, f)) },

  { title: 'Ensembles: position decoding and in-silico ablation',
    files: range('ensemble_', 53, 59).map(f => path.join(FIG, f)) },

  { title: 'Ensembles: trajectories, stability and networks',
    files: [...range('ensemble_', 60, 99),
            ...all.filter(f => /^ensemble_[a-z]/.test(f)).sort()]
           .map(f => path.join(FIG, f)) },

  { title: 'Temporal CCA: strength, direction and sparsity',
    files: fs.readdirSync(TFIG).filter(f => f.startsWith('grid_') && f.endsWith('.png'))
            .sort().map(f => path.join(TFIG, f)) },

  { title: 'Spatial CCA: rerun on the corrected 5 cm cache',
    files: fs.readdirSync(CFIG)
            .filter(f => /committed_partial\.png$/.test(f) && !f.includes('common_units'))
            .sort().map(f => path.join(CFIG, f)) },

  { title: 'RL model: fits, latents and neural encoding',
    files: ['fig_behaviour.png', 'fig_real_lick_profiles.png', 'fig_real_fit_quality.png',
            'fig_example_latents.png', 'fig_latent_recovery.png', 'fig_param_recovery.png',
            'fig_real_latents_value.png', 'fig_real_latents_rpe.png',
            'fig_neural_encoding.png', 'fig_encoding_stats.png', 'fig_encoding_examples.png']
           .filter(f => fs.existsSync(path.join(RFIG, f))).map(f => path.join(RFIG, f)) },

  { title: 'LFP: export integrity, file identity and geometry',
    files: pickIn(LFIG, f => !LFP_SUPERSEDED(f) &&
                    /^(sanity_audit_|signal_identity|lfp_(cohort_overview|spectra|session_integrity|identity_matrix|depth_by_frequency))/.test(f),
                  ['sanity_audit_overview_v2.png', 'sanity_audit_raw_examples_v2.png',
                   'sanity_audit_event_timing.png', 'signal_identity.png',
                   'lfp_cohort_overview_task.png', 'lfp_identity_matrix_task.png']) },

  { title: 'LFP band power across learning: task cohort (16/16, 1212 at full length)',
    files: pickIn(LFIG, f => !LFP_SUPERSEDED(f) && /_task\.png$/.test(f) &&
                    !/^(lfp_cohort_overview|lfp_spectra|lfp_session_integrity|lfp_identity_matrix|lfp_depth_by_frequency)/.test(f),
                  ['lfp_evolution_z_task.png', 'lfp_evolution_speed_task.png',
                   'lfp_evolution_speed_residual_task.png', 'lfp_evolution_fraction_task.png',
                   'lfp_decoding_task.png', 'lfp_reliability_task.png', 'lfp_cca_task.png']) },

  { title: 'LFP band power: yoked Control 1 cohort',
    files: pickIn(LFIG, f => !LFP_SUPERSEDED(f) && /_control\.png$/.test(f) &&
                    !/^(lfp_cohort_overview|lfp_spectra|lfp_session_integrity|lfp_identity_matrix|lfp_depth_by_frequency)/.test(f),
                  ['lfp_evolution_z_control.png', 'lfp_decoding_control.png',
                   'lfp_reliability_control.png', 'lfp_cca_control.png']) },

  { title: 'LFP: task versus yoked control, and the one learning-specific effect',
    files: pickIn(LFIG, /_task_vs_control\.png$/,
                  ['lfp_task_vs_control.png', 'lfp_dls_theta_task_vs_control.png',
                   'lfp_reliability_first20_task_vs_control.png']) },
];

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';            // 13.3 x 7.5 in
const W = 13.3, H = 7.5;

let nFig = 0;
SECTIONS.forEach(sec => {
  if (!sec.files.length) return;

  // --- section divider: single descriptive title on plain white ---
  const s = pres.addSlide();
  s.background = { color: 'FFFFFF' };
  s.addText(sec.title, {
    x: 1.0, y: 2.6, w: W - 2.0, h: 2.3,
    fontSize: 40, bold: true, color: '111111', fontFace: 'Cambria',
    align: 'center', valign: 'middle',
  });

  // --- figure slides: image only, no text of any kind ---
  sec.files.forEach(fp => {
    const fs2 = pres.addSlide();
    fs2.background = { color: 'FFFFFF' };
    // Compute the contain-fit ourselves and pass explicit w/h: pptxgenjs's
    // sizing:{type:'contain'} writes the *box* dimensions onto the shape, so
    // every non-matching aspect ratio ships stretched.
    const { w: iw, h: ih } = pngSize(fp);
    const boxW = W - 0.7, boxH = H - 0.7;
    const scale = Math.min(boxW / iw, boxH / ih);
    const dw = iw * scale, dh = ih * scale;
    fs2.addImage({
      path: fp,
      x: (W - dw) / 2, y: (H - dh) / 2,
      w: dw, h: dh,
    });
    nFig++;
  });
});

pres.writeFile({ fileName: OUT }).then(() => {
  console.log(`wrote ${OUT}`);
  console.log(`sections: ${SECTIONS.filter(s => s.files.length).length}, figure slides: ${nFig}`);
  SECTIONS.filter(s => s.files.length).forEach(s =>
    console.log(`  ${String(s.files.length).padStart(3)}  ${s.title}`));

  // Coverage: a deck that claims "all results" must be able to say what it
  // leaves out. Unused figures are reported by family, not one by one.
  const used = new Set(SECTIONS.flatMap(s => s.files));
  for (const [label, dir] of [['figures/', FIG], ['lfp/figures/', LFIG]]) {
    const unused = fs.readdirSync(dir)
      .filter(f => f.endsWith('.png') && !used.has(path.join(dir, f)));
    if (!unused.length) continue;
    const fams = new Map();
    for (const f of unused) {
      const fam = (f.match(/^([_A-Za-z0-9]+?)(?=_\d+_|_)/) || [null, f])[1];
      fams.set(fam, (fams.get(fam) || 0) + 1);
    }
    const top = [...fams].sort((a, b) => b[1] - a[1]).slice(0, 12);
    console.log(`not in the deck — ${label} ${unused.length} png:`);
    console.log('  ' + top.map(([k, v]) => `${k}(${v})`).join(' ') +
                (fams.size > top.length ? ` +${fams.size - top.length} more families` : ''));
  }
});
