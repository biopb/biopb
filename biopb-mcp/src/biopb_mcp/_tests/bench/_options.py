"""The run options: which cases, in what configuration, how many times.

**One invocation is one configuration.** `--bench-skills` decides whether the
agent is offered the catalog at all and `--bench-responder` decides who answers
it — or whether there is anything left to ask; both are settings on the session
the run happens in, and neither varies within a run. What used to be a 2x2 the
engine iterated is one command per corner, and each writes its own session
directory that says which corner it was.

That is the whole reason there is no `--bench-arms` here any more. An arm was a
*harness configuration the engine chose per case*, which meant a case's kind
decided how much a run cost and the report had to explain a table whose rows
were configured differently from each other. A switch is the same information
with the choice moved to where a person makes it.

Stdlib only, and it imports nothing else in this package. `pytest_addoption` is
answered from the **tests-root** conftest — pytest only calls that hook on the
conftests it loads at startup, so an option declared in `bench/conftest.py`
would be silently absent whenever anyone ran the suite from above it. That
makes this module part of every pytest startup in the repo, including the ones
that never touch a benchmark, and an import chain reaching numpy or a provider
table would put those runs behind it. See `test_report.py` for the check that
the values offered here are the ones the engine can actually honour.

**A flag beats the environment, and there is no dotenv.** `agentbench` reads
`.env` for credentials and model selection, which are facts about a machine. An
option here decides what gets *spent* and what a number means — a case list, a
catalog, a sample count — and a file somebody put in the repo root a month ago
should not be what answers that. An explicit `--bench-skills=false` on the
command line is why a report has no catalog in it; `BIOPB_BENCH_SKILLS` in a
`.env` is not.

**Selecting one case is `-k`**, pytest's own. The parametrization ids are case
labels, so `-k drift-correction`, `-k "drift or flatfield"` and the full
`-k drift-correction/two-channels-one-structural` all work, and there is no
reason for this module to grow a second way to say it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


#: One per option. Kept as data because three things read it — the argparse
#: registration, the environment fallback, and the header line a run prints
#: about itself — and a fourth copy of "the values `--bench-cases` accepts" is
#: exactly the drift this package was merged to stop having.
@dataclass(frozen=True)
class Setting:
    flag: str
    env: str
    values: tuple[str, ...]
    default: str
    help: str

    @property
    def dest(self) -> str:
        return self.flag.lstrip("-").replace("-", "_")


CASES = Setting(
    "--bench-cases",
    "BIOPB_BENCH_CASES",
    ("all", "skills", "tasks"),
    "all",
    "which cases to run: `skills` are the ones making a claim about a served "
    "skill, `tasks` is the complement — every case that asks only whether the "
    "work gets done, including those written alongside a banked skill the "
    "runtime does not serve. Use -k to pick out one case by name",
)

FIXTURES = Setting(
    "--bench-fixtures",
    "BIOPB_BENCH_FIXTURES",
    ("all", "synthetic", "curated"),
    "all",
    "which fixtures to run against: `synthetic` is procedurally built and "
    "always available, `curated` is real data and needs $BIOPB_FIXTURES",
)

#: The catalog switch. `false` is `services.skills_enabled: false` in the
#: session's own config — a real shipped configuration, so the kernel, napari,
#: dask and every library stay exactly as they are and only the curated
#: procedures go. §6's rule: disclose the environment, withhold only the skill.
#:
#: Against a case that names a skill, two runs either side of this is that
#: skill's behavioural delta, and it is the number the whole layer exists to
#: produce. Against a case that names none it asks a fair question too — were
#: the skills helping this at all — it is simply not the one the case was
#: written for.
SKILLS = Setting(
    "--bench-skills",
    "BIOPB_BENCH_SKILLS",
    ("true", "false"),
    "true",
    "whether the agent is offered the skills catalog at all; `false` withholds "
    "it and nothing else, which is the ablation half of a skill's delta",
)

#: Who answers when the agent asks — and, for one value, whether there is
#: anything left to ask. Three, and none of them is a straw man.
#:
#: `silent` is the control condition: "I don't know" is what a real user says
#: about half the metadata they are asked for, and `calibrated-measurements`
#: specifies that branch explicitly. A run against it **must fail** a case whose
#: fixture withholds a fact — if it does not, that asymmetry is decorative and
#: the case measures something else. That pair of runs is a claim about the
#: *fixture*, not about a skill.
#:
#: `briefed` varies the other thing. The persona's facts go into the task prompt
#: at handover and the respondent adds nothing after, so the run has the whole
#: of the information and none of the conversation. Against `model` that is the
#: **cost of having to elicit** — a delta the other two cannot separate from the
#: value of the fact itself, since they differ in the information as well as in
#: the exchange. Read the three together: `silent` says whether the fact was
#: obtainable from the pixels, `briefed` says what obtaining it *by asking* was
#: worth over being handed it.
RESPONDER = Setting(
    "--bench-responder",
    "BIOPB_BENCH_RESPONDER",
    ("model", "silent", "briefed"),
    "model",
    "who answers the agent's questions: `model` plays the case's persona, "
    "`silent` answers nothing, which is what says whether the withheld fact "
    "was obtainable from the pixels, and `briefed` puts every fact the persona "
    "holds into the task prompt up front and answers nothing after, which is "
    "the same information with the asking taken out",
)

SETTINGS = (CASES, FIXTURES, SKILLS, RESPONDER)

#: How many times to run the case. One unless asked for more.
#:
#: A single sample is the right default for iterating on a case; it is not a
#: measurement. Raise it when a number is going to be quoted, because the
#: spread between runs of the same case is routinely larger than the difference
#: anyone is trying to read off it. With one configuration per invocation this
#: is the only axis a single run varies.
SAMPLES_FLAG = "--bench-samples"
SAMPLES_ENV = "BIOPB_BENCH_SAMPLES"
SAMPLES_DEST = "bench_samples"


class BadOption(ValueError):
    """A run option that cannot be honoured. Raised rather than defaulted."""


@dataclass(frozen=True)
class Options:
    """One run's answers, and so one run's whole configuration.

    Plain values, so nothing here can fail late: each was checked against its
    own vocabulary at resolve time. This object is what `session.json` records,
    and it is the only thing that makes two report directories comparable.
    """

    cases: str = CASES.default
    fixtures: str = FIXTURES.default
    skills: bool = True
    responder: str = RESPONDER.default
    samples: int = 1

    @property
    def filtered(self) -> bool:
        """Whether the *case list* was narrowed. Decides if a run has to say so.

        Only the two selection options count. `skills` and `responder` change
        what a run measures rather than how much of the catalogue it covers,
        and both are already in every report header and in `session.json`.
        """
        return self.cases != CASES.default or self.fixtures != FIXTURES.default

    @property
    def configuration(self) -> str:
        """The session's harness configuration, short enough for a heading."""
        return f"skills={'on' if self.skills else 'off'} responder={self.responder}"

    def describe(self) -> str:
        """One line, for the report and the terminal. Every option, including
        the ones left alone — a header that lists only what was changed cannot
        be read as a record of what was run."""
        return (
            f"cases={self.cases} fixtures={self.fixtures} "
            f"skills={str(self.skills).lower()} responder={self.responder} "
            f"samples={self.samples}"
        )

    def as_json(self) -> dict:
        """What `session.json` carries. Keys are the option names, so a reader
        who has seen `pytest -h` needs nothing else to interpret them."""
        return {
            "cases": self.cases,
            "fixtures": self.fixtures,
            "skills": self.skills,
            "responder": self.responder,
            "samples": self.samples,
        }


def add_options(parser) -> None:
    """Register the flags. Called from the tests-root conftest, once."""
    group = parser.getgroup("bench", "the agent benchmark (-m bench)")
    for setting in SETTINGS:
        group.addoption(
            setting.flag,
            dest=setting.dest,
            default=None,
            choices=setting.values,
            help=f"{setting.help} [${setting.env}, default: {setting.default}]",
        )
    group.addoption(
        SAMPLES_FLAG,
        dest=SAMPLES_DEST,
        default=None,
        metavar="N",
        help=(
            "how many times to run each case; one sample is not a measurement "
            f"[${SAMPLES_ENV}, default: 1]"
        ),
    )


def _chosen(config, setting: Setting) -> str:
    flag = config.getoption(setting.dest, None) if config is not None else None
    if flag:
        return flag
    raw = os.environ.get(setting.env, "").strip().lower()
    if not raw:
        return setting.default
    if raw not in setting.values:
        raise BadOption(
            f"{setting.env}={raw!r} is not one of {list(setting.values)}. "
            f"Values are not guessed at here: these options decide what a run "
            f"spends and what its number means, and a typo that quietly ran "
            f"the other configuration is not visible in the report."
        )
    return raw


def _samples(config) -> int:
    raw = config.getoption(SAMPLES_DEST, None) if config is not None else None
    if raw is None:
        raw = os.environ.get(SAMPLES_ENV, "").strip()
    if not raw:
        return 1
    try:
        count = int(raw)
    except ValueError as exc:
        raise BadOption(f"{SAMPLES_FLAG}={raw!r} is not a number") from exc
    if count < 1:
        # Rather than clamping to 1. Asking for zero samples is not a request
        # for one; it is someone expecting a run they will not get, and a
        # silently-corrected count reads afterwards exactly like a real result.
        raise BadOption(f"{SAMPLES_FLAG}={count} — a run is at least one sample")
    return count


def resolve(config=None) -> Options:
    """This run's options: the flag, else the environment, else the default."""
    return Options(
        cases=_chosen(config, CASES),
        fixtures=_chosen(config, FIXTURES),
        skills=_chosen(config, SKILLS) == "true",
        responder=_chosen(config, RESPONDER),
        samples=_samples(config),
    )
