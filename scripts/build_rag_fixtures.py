"""Generate the labeled RAG evaluation set (synthetic, deterministic).

30 drugs across 10 fictional tickers. Per drug:
  * 2 RELEVANT docs stating its specific risk fact (Tier-2 10-K risk factor, Tier-3 abstract)
  * 10 DISTRACTOR docs about the same drug (5 business, 5 generic risk boilerplate)
Plus: a near-identical risk doc for a drug of a DIFFERENT ticker (semantic-bleed bait), and
one renamed program whose older 10-K mentions only its previous code.

Outputs data/fixtures/rag/{synonyms.yaml, corpus.json, queries.json}.
Run:  python scripts/build_rag_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "rag"

TICKERS = ["QONC", "QNEU", "QCAR", "QIMM", "QGEN", "QRAR", "QDER", "QOPH", "QHEM", "QVAX"]
GENERIC_STEMS = ["zolar", "veno", "trima", "palu", "quen", "ribo", "sava", "tolu", "umbra",
                 "xeli"]
SUFFIXES = ["tinib", "mab", "stat"]

# (short label, specific query phrase, 10-K style sentence, abstract style sentence)
RISKS: list[tuple[str, str, str, str]] = [
    ("hepatotoxicity", "liver injury",
     "elevations of ALT and AST above five times the upper limit of normal occurred in 4% of "
     "patients, and two cases met Hy's law criteria for drug-induced liver injury",
     "hepatic adverse events including grade 3 transaminase elevations were observed, "
     "suggesting a signal of drug-induced liver injury"),
    ("QT prolongation", "cardiac QT interval",
     "a thorough QT study showed mean QTc prolongation of 18 milliseconds, raising the risk of "
     "torsades de pointes and possible labeling restrictions",
     "dose-dependent QTc interval prolongation was detected on electrocardiogram monitoring"),
    ("partial clinical hold", "FDA clinical hold",
     "the FDA placed a partial clinical hold on the program after a patient death, and dosing "
     "of new patients is suspended until we submit a complete response to the hold",
     "enrollment was paused under a partial clinical hold following a fatal adverse event"),
    ("manufacturing CRL", "manufacturing deficiencies at the plant",
     "the FDA cited observations on Form 483 at our contract manufacturer, and unresolved "
     "deficiencies could result in a complete response letter for the application",
     "the inspection of the fill-finish facility identified sterility assurance deficiencies"),
    ("neutropenia", "low neutrophil counts",
     "dose-limiting grade 4 neutropenia occurred in 22% of patients, requiring dose reductions "
     "and growth factor support",
     "severe neutropenia was the most common dose-limiting toxicity in the expansion cohort"),
    ("cytokine release syndrome", "cytokine release syndrome",
     "cytokine release syndrome of grade 3 or higher occurred in 9% of patients, and one "
     "patient required intensive care, which may require a REMS program",
     "high-grade cytokine release syndrome events were managed with tocilizumab"),
    ("suicidality signal", "suicidal ideation",
     "an imbalance in suicidal ideation and behavior was observed versus placebo, which may "
     "lead to a boxed warning",
     "the Columbia Suicide Severity Rating Scale showed more events of suicidal ideation in "
     "the treatment arm"),
    ("missed secondary endpoint", "secondary endpoints",
     "the trial did not show statistical significance on the key secondary endpoint of "
     "overall survival, which regulators may weigh heavily",
     "overall survival, a key secondary endpoint, was not statistically different between arms"),
    ("single-arm data", "lack of a randomized control",
     "the application relies on a single-arm study without a randomized comparator, and the "
     "FDA has stated a confirmatory randomized trial is required",
     "efficacy was evaluated in an uncontrolled single-arm design, limiting causal inference"),
    ("immunogenicity", "anti-drug antibodies",
     "neutralizing anti-drug antibodies developed in 31% of patients and were associated with "
     "loss of efficacy",
     "treatment-emergent neutralizing antibodies reduced drug exposure in a third of subjects"),
    ("retinal toxicity", "eye and retinal damage",
     "retinal pigment epithelium changes were seen in long-term animal studies and in two "
     "patients, requiring ophthalmologic monitoring",
     "ocular examinations revealed retinal changes consistent with drug-related toxicity"),
    ("thrombocytopenia", "low platelet counts",
     "grade 3 thrombocytopenia with bleeding events occurred in 7% of patients",
     "platelet counts fell below 50,000 per microliter in a subset of patients with "
     "associated hemorrhage"),
    ("bone marrow suppression", "myelosuppression",
     "prolonged myelosuppression lasting more than 28 days occurred in 5% of patients",
     "delayed hematologic recovery indicated persistent marrow suppression"),
    ("nephrotoxicity", "kidney damage",
     "acute kidney injury with serum creatinine doubling occurred in 3% of patients",
     "renal adverse events included acute kidney injury and proteinuria"),
    ("pancreatitis", "pancreatitis",
     "cases of acute pancreatitis, including one fatal case, were reported in the open-label "
     "extension",
     "elevated lipase and clinical pancreatitis were observed during treatment"),
    ("off-target gene editing", "off-target edits",
     "genome sequencing detected off-target editing at low frequency, the long-term "
     "consequences of which are unknown",
     "unintended edits at predicted off-target loci were identified by deep sequencing"),
    ("insertional oncogenesis", "vector integration and cancer",
     "a patient treated with our lentiviral product developed myelodysplastic syndrome, and "
     "vector integration is being investigated",
     "a case of clonal expansion with vector integration near an oncogene was reported"),
    ("injection-site necrosis", "injection site reactions",
     "injection-site necrosis requiring debridement occurred in four patients",
     "severe local injection site reactions including necrosis were documented"),
    ("hypersensitivity", "anaphylaxis",
     "anaphylaxis occurred in 1.2% of infusions, including two life-threatening events",
     "infusion-related hypersensitivity reactions included anaphylaxis requiring epinephrine"),
    ("liver enzyme drug interaction", "CYP3A4 drug interactions",
     "the drug is a strong CYP3A4 inhibitor, and coadministration with common medications "
     "raised exposure fivefold, which may restrict the label",
     "pharmacokinetic studies showed clinically significant CYP3A4-mediated interactions"),
    ("placebo response", "high placebo response",
     "a high placebo response rate of 41% in the pivotal study narrowed the treatment effect "
     "below the clinically meaningful threshold",
     "the placebo arm improved substantially, attenuating the observed effect size"),
    ("dosing errors", "medication errors",
     "the complex weight-based titration scheme led to dosing errors in 6% of patients",
     "medication errors related to titration complexity were frequent in the study"),
    ("cardiomyopathy", "heart failure",
     "decreases in left ventricular ejection fraction of more than 10 points occurred in 8% of "
     "patients",
     "echocardiography revealed reductions in ejection fraction consistent with cardiomyopathy"),
    ("interstitial lung disease", "lung inflammation",
     "interstitial lung disease or pneumonitis occurred in 5% of patients, including one "
     "death",
     "pneumonitis and interstitial lung disease were reported as serious adverse events"),
    ("hyperglycemia", "high blood sugar",
     "grade 3 hyperglycemia occurred in 14% of patients and required insulin therapy",
     "treatment-emergent hyperglycemia was common and managed with metformin or insulin"),
    ("peripheral neuropathy", "nerve damage",
     "peripheral sensory neuropathy led to discontinuation in 11% of patients",
     "cumulative dose-dependent peripheral neuropathy was observed"),
    ("embryo-fetal toxicity", "birth defects",
     "animal reproduction studies showed embryo-fetal toxicity, and the label may require a "
     "pregnancy prevention program",
     "teratogenic effects were observed in rat and rabbit embryo-fetal development studies"),
    ("supply of starting material", "single-source raw material",
     "a key starting material is available from a single supplier whose facility recently "
     "received an FDA warning letter",
     "the synthesis depends on an intermediate supplied by one qualified vendor"),
    ("PDUFA delay", "regulatory review extension",
     "the FDA extended the review goal date by three months after we submitted a major "
     "amendment to the application",
     "the regulatory review was extended following submission of additional analyses"),
    ("rare fatal infection", "opportunistic infections",
     "two patients developed progressive multifocal leukoencephalopathy, a rare and often "
     "fatal brain infection",
     "opportunistic infections including PML were reported in the long-term cohort"),
]

BUSINESS = [
    "We are building a specialty sales force of approximately 60 representatives to support "
    "the commercial launch of {name} in the United States.",
    "Our manufacturing partner produces {name} drug substance at a facility with capacity "
    "sufficient for the first three years of anticipated demand.",
    "We hold issued composition-of-matter patents covering {name} that expire in 2039, "
    "excluding any patent term extension.",
    "We entered into a collaboration agreement for {name} in Japan that provides an upfront "
    "payment and potential milestone payments.",
    "Our medical affairs team presented health economics data for {name} at a payer "
    "conference to support future reimbursement discussions.",
]

BOILERPLATE = [
    "We face substantial competition, and competitors may develop products similar to {name} "
    "that are approved earlier or are more effective.",
    "Coverage and reimbursement for {name} may be limited, which could reduce demand even if "
    "{name} is approved.",
    "We depend on key personnel, and losing members of our management team could delay the "
    "development of {name}.",
    "Our stock price is volatile and may fluctuate significantly in response to announcements "
    "about {name} or our competitors.",
    "Changes in healthcare legislation and drug pricing policy could adversely affect sales "
    "of {name}.",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entities: dict[str, dict[str, object]] = {}
    docs: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []

    for i, (label, phrase, k10, abstract) in enumerate(RISKS):
        ticker = TICKERS[i // 3]
        j = i % 3
        code = f"{ticker[1:]}-{101 + j * 100}"
        generic = f"{GENERIC_STEMS[i // 3]}{SUFFIXES[j]}"
        eid = f"ENT-{ticker}-{101 + j * 100}"
        names = [code, generic]
        entity: dict[str, object] = {"ticker": ticker, "names": names}
        if i == 0:  # renamed program: older filings used a different code
            entity["previous_codes"] = ["ZX-7"]
        entities[eid] = entity

        relevant_keys = [f"{eid}:risk10k", f"{eid}:riskabs"]
        docs.append({"key": relevant_keys[0], "ticker": ticker, "entity_id": eid, "tier": 2,
                     "published_at": "2025-03-01T00:00:00+00:00", "doc_type": "10-K-1A",
                     "url": f"https://example.org/{ticker}/10k#{code}",
                     "title": f"{ticker} 10-K risk factor: {label}",
                     "text": f"Risks related to {generic} ({code}). In our clinical program for "
                             f"{generic}, {k10}."})
        docs.append({"key": relevant_keys[1], "ticker": ticker, "entity_id": eid, "tier": 3,
                     "published_at": "2025-06-01T00:00:00+00:00", "doc_type": "pubmed",
                     "url": f"https://example.org/pubmed/{eid}",
                     "title": f"Safety of {generic} in a phase 2 trial",
                     "text": f"In this phase 2 study of {code} ({generic}), {abstract}."})
        for n, tmpl in enumerate(BUSINESS + BOILERPLATE):
            name = generic if n % 2 else code
            docs.append({"key": f"{eid}:d{n}", "ticker": ticker, "entity_id": eid,
                         "tier": 2 if n >= len(BUSINESS) else 3,
                         "published_at": "2025-04-01T00:00:00+00:00",
                         "doc_type": "10-K-1A" if n >= len(BUSINESS) else "press",
                         "url": f"https://example.org/{ticker}/doc{n}#{code}",
                         "title": f"{ticker} disclosure {n}",
                         "text": tmpl.format(name=name)})
        # Semantic-bleed bait: same risk text, different ticker, different drug.
        bait_ticker = TICKERS[(i // 3 + 5) % len(TICKERS)]
        docs.append({"key": f"BAIT:{eid}", "ticker": bait_ticker,
                     "entity_id": f"ENT-{bait_ticker}-COMPANY", "tier": 2,
                     "published_at": "2025-03-01T00:00:00+00:00", "doc_type": "10-K-1A",
                     "url": f"https://example.org/{bait_ticker}/bait{i}",
                     "title": f"{bait_ticker} risk factor",
                     "text": f"Risks related to {generic} ({code}). In our clinical program "
                             f"for {generic}, {k10}."})
        queries.append({"id": f"q{i}-specific", "ticker": ticker, "drug": generic,
                        "query": f"Is there a {phrase} risk with {generic}?",
                        "relevant": relevant_keys})
        queries.append({"id": f"q{i}-generic", "ticker": ticker, "drug": code,
                        "query": f"What are the specific safety and regulatory problems "
                                 f"reported for {code} in clinical trials?",
                        "relevant": relevant_keys})

    # Renamed program: an older 10-K that only knows the previous code ZX-7.
    first = next(iter(entities))
    docs.append({"key": "RENAMED:old10k", "ticker": TICKERS[0], "entity_id": None, "tier": 2,
                 "published_at": "2023-03-01T00:00:00+00:00", "doc_type": "10-K-1A",
                 "url": "https://example.org/QONC/10k-2022",
                 "title": "QONC 2022 10-K risk factor",
                 "text": "Risks related to ZX-7. Two patients receiving ZX-7 developed "
                         "drug-induced liver injury meeting Hy's law, and further hepatic "
                         "safety monitoring is required."})
    queries.append({"id": "q-renamed", "ticker": TICKERS[0], "drug": "ONC-101",
                    "query": "What liver safety findings were reported for ONC-101 under "
                             "its earlier development code?",
                    "relevant": ["RENAMED:old10k"]})

    (OUT / "synonyms.yaml").write_text(
        "# Fixture ontology for RAG tests (synthetic drugs).\n"
        + yaml.safe_dump({"entities": entities}, sort_keys=True), encoding="utf-8")
    (OUT / "corpus.json").write_text(json.dumps(docs, indent=1) + "\n", encoding="utf-8")
    (OUT / "queries.json").write_text(json.dumps(queries, indent=1) + "\n", encoding="utf-8")
    print(f"{len(entities)} entities, {len(docs)} docs, {len(queries)} queries; first={first}")


if __name__ == "__main__":
    main()
