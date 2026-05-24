"""
Post-hoc physics plausibility checks for CICIoT2023 flow features.

These checks are intentionally separate from the G1-G8 domain validator:
they measure cross-feature physical consistency after inverse scaling rather
than structural/statistical validity enforced by the decoder.

Notes on dataset semantics used here
-----------------------------------
- ``Tot size`` is ambiguous in the paper and, in this project's processed
  schema, is an exact duplicate of ``AVG``. Because that makes its physical
  meaning unclear, P3 is intentionally omitted from aggregate reporting rather
  than guessed.
- P1, P6, P7, and P8 were prototyped and calibrated, but removed from the
  active rule set because clean raw dataset pass rates were too low for use as
  fidelity metrics without redefining them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _RuleSpec:
    name: str
    implemented: bool


class PhysicsValidator:
    """Secondary raw-space plausibility checks for inverse-transformed samples."""

    PROTOCOL_MIN_HEADER = {
        0: 0.0,   # HOPOPT
        1: 8.0,   # ICMP
        2: 8.0,   # IGMP
        6: 20.0,  # TCP
        17: 8.0,  # UDP
        47: 4.0,  # GRE
    }

    FLAG_PAIRS = (
        ("syn_flag_number", "syn_count"),
        ("ack_flag_number", "ack_count"),
        ("fin_flag_number", "fin_count"),
        ("rst_flag_number", "rst_count"),
    )

    TCP_FLAG_INDICATORS = (
        "syn_flag_number",
        "ack_flag_number",
        "fin_flag_number",
        "rst_flag_number",
        "psh_flag_number",
        "ece_flag_number",
        "cwr_flag_number",
    )

    TCP_FLAG_COUNTS = (
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
    )

    def __init__(
        self,
        feature_names: list[str],
        rtol: float = 0.05,
        eps: float = 1e-6,
        count_eps: float = 0.5,
        std_eps: float = 1e-3,
    ) -> None:
        self.feature_names = list(feature_names)
        self.idx = {name: i for i, name in enumerate(self.feature_names)}
        self.rtol = float(rtol)
        self.eps = float(eps)
        self.count_eps = float(count_eps)
        self.std_eps = float(std_eps)
        self.var_eps = 1e-6

        self._rules = (
            _RuleSpec("P2", self._has_all("Tot sum", "Number", "AVG")),
            _RuleSpec("P3", False),
            _RuleSpec("P4", self._has_all("Std", "Min", "Max")),
            _RuleSpec("P5", self._has_all("Number", "Std", "Variance", "Min", "AVG", "Max")),
        )

    def _has_all(self, *names: str) -> bool:
        return all(name in self.idx for name in names)

    def _col(self, x_raw: np.ndarray, name: str) -> np.ndarray:
        return np.asarray(x_raw[:, self.idx[name]], dtype=np.float64)

    def _true_mask(self, x_raw: np.ndarray) -> np.ndarray:
        return np.ones(int(x_raw.shape[0]), dtype=bool)

    def _rounded_indicator(self, values: np.ndarray) -> np.ndarray:
        return np.rint(values).astype(int)

    def check_p1_rate_consistency(self, x_raw: np.ndarray) -> np.ndarray:
        """Return True when ``Rate`` is consistent with per-packet ``IAT``.

        Vacuous-true handling:
        - ``Number <= 1`` returns True.
        - ``IAT <= eps`` returns True.
        """
        if not self._has_all("Rate", "IAT", "Number"):
            return self._true_mask(x_raw)

        rate = self._col(x_raw, "Rate")
        iat = self._col(x_raw, "IAT")
        number = self._col(x_raw, "Number")

        vacuous = (number <= 1.0) | (iat <= self.eps)
        expected_rate = 1.0 / np.maximum(iat, self.eps)
        rel_err = np.abs(rate - expected_rate) / np.maximum(np.abs(rate), self.eps)
        return vacuous | (rel_err < self.rtol)

    def check_p2_total_bytes(self, x_raw: np.ndarray) -> np.ndarray:
        """Return True when ``Tot sum`` matches ``Number * AVG`` within tolerance."""
        if not self._has_all("Tot sum", "Number", "AVG"):
            return self._true_mask(x_raw)

        total = self._col(x_raw, "Tot sum")
        number = self._col(x_raw, "Number")
        avg = self._col(x_raw, "AVG")

        rel_err = np.abs(total - (number * avg)) / np.maximum(np.abs(total), self.eps)
        return rel_err < self.rtol

    def check_p3_total_size(self, x_raw: np.ndarray) -> np.ndarray:
        """Return all-True because P3 is intentionally omitted for this schema."""
        return self._true_mask(x_raw)

    def check_p4_std_magnitude(self, x_raw: np.ndarray) -> np.ndarray:
        """Return True when ``Std`` does not exceed half the min-max range."""
        if not self._has_all("Std", "Min", "Max"):
            return self._true_mask(x_raw)

        std = self._col(x_raw, "Std")
        min_ = self._col(x_raw, "Min")
        max_ = self._col(x_raw, "Max")
        bound = ((max_ - min_) / 2.0) * (1.0 + self.rtol)
        return std <= bound

    def check_p5_single_packet(self, x_raw: np.ndarray) -> np.ndarray:
        """Return True for non-singleton flows, else enforce zero-variance shape.

        Vacuous-true handling:
        - ``Number > 1`` returns True.
        """
        if not self._has_all("Number", "Std", "Variance", "Min", "AVG", "Max"):
            return self._true_mask(x_raw)

        number = self._col(x_raw, "Number")
        std = self._col(x_raw, "Std")
        var = self._col(x_raw, "Variance")
        min_ = self._col(x_raw, "Min")
        avg = self._col(x_raw, "AVG")
        max_ = self._col(x_raw, "Max")

        singleton = np.isclose(number, 1.0, atol=self.eps)
        rel_span = np.abs(min_ - max_) / np.maximum(np.abs(avg), self.eps)
        checks = (std < self.std_eps) & (var < self.var_eps) & (rel_span < self.rtol)
        return (~singleton) | checks

    def check_p6_header_length(self, x_raw: np.ndarray) -> np.ndarray:
        """Return True when header length clears the protocol minimum."""
        if not self._has_all("Header_Length", "Protocol Type"):
            return self._true_mask(x_raw)

        header = self._col(x_raw, "Header_Length")
        proto = np.rint(self._col(x_raw, "Protocol Type")).astype(int)

        min_header = np.zeros_like(header)
        applicable = np.zeros_like(header, dtype=bool)
        for proto_id, threshold in self.PROTOCOL_MIN_HEADER.items():
            mask = proto == proto_id
            applicable |= mask
            min_header[mask] = threshold

        passes = header >= (min_header * (1.0 - self.rtol))
        return (~applicable) | passes

    def check_p7_flag_indicator_count(self, x_raw: np.ndarray) -> np.ndarray:
        """Return True when each TCP flag indicator agrees with its count field."""
        if not self._has_all(*(x for pair in self.FLAG_PAIRS for x in pair)):
            return self._true_mask(x_raw)

        passed = self._true_mask(x_raw)
        for indicator_name, count_name in self.FLAG_PAIRS:
            indicator = self._rounded_indicator(self._col(x_raw, indicator_name)) == 1
            count_present = self._col(x_raw, count_name) > self.count_eps
            passed &= indicator == count_present
        return passed

    def check_p8_non_tcp_no_tcp_flags(self, x_raw: np.ndarray) -> np.ndarray:
        """Return True for TCP rows, else require all TCP-specific flags to be zero.

        Vacuous-true handling:
        - ``Protocol Type == 6`` returns True.
        """
        if not self._has_all("Protocol Type", *self.TCP_FLAG_INDICATORS, *self.TCP_FLAG_COUNTS):
            return self._true_mask(x_raw)

        proto = np.rint(self._col(x_raw, "Protocol Type")).astype(int)
        tcp_rows = proto == 6

        ind_zero = self._true_mask(x_raw)
        for name in self.TCP_FLAG_INDICATORS:
            ind_zero &= np.abs(self._col(x_raw, name)) <= self.eps

        count_zero = self._true_mask(x_raw)
        for name in self.TCP_FLAG_COUNTS:
            count_zero &= self._col(x_raw, name) < self.count_eps

        return tcp_rows | (ind_zero & count_zero)

    def validate_batch(self, x_raw: np.ndarray) -> dict:
        """Validate a raw-space batch and return per-rule and per-sample results."""
        x_raw = np.asarray(x_raw, dtype=np.float64)
        if x_raw.ndim != 2:
            raise ValueError(f"x_raw must be 2D, got shape {x_raw.shape}")

        rule_results = {
            "P2": self.check_p2_total_bytes(x_raw),
            "P4": self.check_p4_std_magnitude(x_raw),
            "P5": self.check_p5_single_packet(x_raw),
        }

        implemented_rules = [spec.name for spec in self._rules if spec.implemented]
        per_sample_pass_mask = np.stack([rule_results[name] for name in implemented_rules], axis=1)
        per_rule_pass_rate = {
            name: float(rule_results[name].mean())
            for name in implemented_rules
        }

        return {
            "per_rule_pass_rate": per_rule_pass_rate,
            "all_rules_pass_rate": float(per_sample_pass_mask.all(axis=1).mean()),
            "n_samples": int(x_raw.shape[0]),
            "per_sample_pass_mask": per_sample_pass_mask,
            "rules_implemented": implemented_rules,
            "rules_omitted": ["P1", "P3", "P6", "P7", "P8"],
            "rule_notes": {
                "P3": (
                    "Omitted because Tot size is ambiguous in the source documentation "
                    "and is an exact duplicate of AVG in this processed schema."
                ),
                "P1": "Removed because clean raw dataset pass rate was below threshold.",
                "P6": "Removed because clean raw dataset pass rate was below threshold.",
                "P7": "Removed because clean raw dataset pass rate was below threshold.",
                "P8": "Removed because clean raw dataset pass rate was below threshold.",
            },
        }
