"""Lu's small Matplotlib style/export helper; no statistical estimation."""
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import seaborn as sns

STYLE_PATH = Path(__file__).resolve().parents[1] / 'lu.mplstyle'
DEFAULTS = mpl.rc_params_from_file(STYLE_PATH, use_default_template=False)


@contextmanager
def plot_context(rc=None):
    """Apply normal talk context and Lu's defaults locally; task overrides last."""
    settings = dict(sns.plotting_context('talk'))
    settings.update(DEFAULTS)
    if rc:
        settings.update(rc)
    with mpl.rc_context(settings):
        yield


def new_panel(width=None, height=None, **kwargs):
    """Create a single panel inside plot_context; dimensions are inches."""
    default_width, default_height = mpl.rcParams['figure.figsize']
    return plt.subplots(figsize=(default_width if width is None else width,
                                 default_height if height is None else height), **kwargs)


def _legend_entries(fig):
    handles, labels = [], []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            hs = legend.legend_handles
            ls = [text.get_text() for text in legend.get_texts()]
        else:
            hs, ls = ax.get_legend_handles_labels()
        handles.extend(hs)
        labels.extend(ls)
    for legend in fig.legends:
        handles.extend(legend.legend_handles)
        labels.extend(text.get_text() for text in legend.get_texts())
    # Shared labels must represent identical encodings. Supply explicit entries
    # for intentionally repeated labels or custom legend handlers.
    unique = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith('_'):
            unique.setdefault(label, handle)
    return list(unique.values()), list(unique)


def _save_legend(handles, labels, path, dpi, ncol, legend_kwargs):
    fig = Figure(figsize=(3.5, 3.5), dpi=dpi, facecolor='white')
    FigureCanvasAgg(fig)
    legend = fig.legend(handles, labels, loc='center', ncol=ncol, **legend_kwargs)
    fig.canvas.draw()
    bounds = legend.get_window_extent(fig.canvas.get_renderer())
    fig.set_size_inches(bounds.width / dpi + .16, bounds.height / dpi + .16)
    fig.canvas.draw()
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({'savefig.bbox': None}):
        fig.savefig(path, dpi=dpi, format=path.suffix[1:], bbox_inches=None,
                    facecolor='white', transparent=False)


def save_panel(fig, path, *, caption, uncertainty, legend=True,
               legend_path=None, legend_entries=None, legend_ncol=1,
               legend_kwargs=None, dpi=None, output_format=None):
    """Save a panel, separate legend if needed, and a standalone caption note.

    Call inside plot_context after populating one panel. The helper cannot judge
    whether a CI is meaningful or correct: uncertainty must explain the actual
    interval or why none applies. Explicit output_format/dpi support user-requested
    task overrides; a filename alone never silently enables another format.
    """
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError('A standalone caption is required.')
    if not isinstance(uncertainty, str) or not uncertainty.strip():
        raise ValueError('An uncertainty definition or non-applicability explanation is required.')
    fmt = output_format or DEFAULTS['savefig.format']
    path = Path(path)
    if path.suffix and path.suffix.lower() != '.' + fmt:
        raise ValueError('Filename format differs from the requested format; pass output_format explicitly.')
    if not path.suffix:
        path = path.with_suffix('.' + fmt)
    dpi = DEFAULTS['savefig.dpi'] if dpi is None else dpi
    entries = _legend_entries(fig) if legend_entries is None else legend_entries
    handles, labels = list(entries[0]), list(entries[1])
    if len(handles) != len(labels):
        raise ValueError('Legend handles and labels must have equal lengths.')
    legend_output = None
    if legend and handles:
        # Even an explicitly requested vector panel keeps a PNG legend by default.
        legend_output = Path(legend_path) if legend_path else path.with_name(path.stem + '-legend.png')
        if legend_output.suffix.lower() != '.png' or legend_output.resolve() == path.resolve():
            raise ValueError('The legend must have a distinct PNG path.')
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for existing in list(fig.legends):
        existing.remove()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None alone can inherit savefig.bbox='tight'. Reset both.
    with mpl.rc_context({'savefig.bbox': None}):
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches=None,
                    facecolor='white', transparent=False)
    if legend_output is not None:
        _save_legend(handles, labels, legend_output, dpi, legend_ncol, legend_kwargs or {})
    note = path.with_name(path.stem + '.caption.md')
    content = caption.strip() + '\n\nUncertainty: ' + uncertainty.strip() + '\n'
    if legend_output:
        content += '\nLegend: ' + str(legend_output.name) + '\n'
    note.write_text(content, encoding='utf-8')
    return {'panel': path, 'legend': legend_output, 'caption': note}
