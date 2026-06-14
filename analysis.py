"""Utilities for loading and organizing UR5 degradation experiment data.

The raw files in ``data/`` are stored as line-delimited Python literals rather
than conventional CSV tables. Each row contains one timestamp and grouped
measurements for the UR5 controller state. The column names are stored in the
separate ``ur5testresult_header.xlsx`` workbook.

This module provides a small loader class that:

* reads the header workbook
* parses every experiment file under ``data/``
* extracts experiment conditions from the file name
* flattens each record into a tabular ``pandas.DataFrame``
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class ExperimentCondition:
	"""Metadata parsed from a single experiment file name."""

	start_condition: str
	speed: str
	payload_lb: float
	replicate: int
	file_name: str

	@property
	def condition_key(self) -> str:
		return f"{self.start_condition}_{self.speed}_payload{self.payload_lb:g}lb"


class UR5DegradationAnalyzer:
	"""Load and organize the UR5 accelerated-aging experiment results."""

	# Nominal standard-DH dimensions for the original UR5 (CB-series), in metres.
	# The raw controller joint positions are recorded in degrees.
	UR5_DH_A = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
	UR5_DH_D = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
	UR5_DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

	FILE_PATTERN = re.compile(
		r"^ur5testresult_(?:(coldstart)_)?(fullspeed|halfspeed)_payload([0-9.]+)lb_(\d+)\.csv$"
	)

	def __init__(
		self,
		data_dir: Optional[Path | str] = None,
		tcp_offset_m: Optional[Sequence[float]] = None,
	) -> None:
		self.project_root = Path(__file__).resolve().parent
		self.data_dir = Path(data_dir) if data_dir is not None else self.project_root / "data"
		self.tcp_offset_m = np.asarray(
			tcp_offset_m if tcp_offset_m is not None else [0.0, 0.0, 0.0],
			dtype=float,
		)
		if self.tcp_offset_m.shape != (3,):
			raise ValueError("tcp_offset_m must contain exactly three values: [x, y, z]")
		self.header_workbook = self.data_dir / "ur5testresult_header.xlsx"
		self.header_columns = self._load_header_columns()
		self.header_details = self._load_header_details()
		self.data_frame: Optional[pd.DataFrame] = None
		self.cartesian_error_data_frame: Optional[pd.DataFrame] = None

	def _load_header_columns(self) -> List[str]:
		"""Read the column names from the header workbook."""

		header_df = pd.read_excel(self.header_workbook, sheet_name="header", header=None)
		first_row = header_df.iloc[0].tolist()
		columns = [str(value).strip() for value in first_row if pd.notna(value) and str(value).strip()]
		if not columns:
			raise ValueError(f"No header columns found in {self.header_workbook}")
		return columns

	def _load_header_details(self) -> pd.DataFrame:
		"""Load the descriptive metadata sheet for later reference."""

		return pd.read_excel(self.header_workbook, sheet_name="header details")

	@staticmethod
	def _flatten_record(record: Sequence[object]) -> List[object]:
		"""Flatten the grouped record structure into a single list of values."""

		flattened: List[object] = []
		for value in record:
			if isinstance(value, (list, tuple)):
				flattened.extend(value)
			else:
				flattened.append(value)
		return flattened

	def _parse_condition(self, file_path: Path) -> ExperimentCondition:
		match = self.FILE_PATTERN.match(file_path.name)
		if match is None:
			raise ValueError(f"Unexpected file name format: {file_path.name}")

		start_condition = match.group(1) or "normalstart"
		speed = match.group(2)
		payload_lb = float(match.group(3))
		replicate = int(match.group(4))

		return ExperimentCondition(
			start_condition=start_condition,
			speed=speed,
			payload_lb=payload_lb,
			replicate=replicate,
			file_name=file_path.name,
		)

	def _iter_csv_files(self) -> List[Path]:
		if not self.data_dir.exists():
			raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

		return sorted(
			path
			for path in self.data_dir.glob("ur5testresult*.csv")
			if path.is_file()
		)

	def _parse_csv_file(self, file_path: Path) -> pd.DataFrame:
		"""Parse one raw experiment file into a flat data frame."""

		condition = self._parse_condition(file_path)
		rows: List[List[object]] = []

		with file_path.open("r", encoding="utf-8") as handle:
			for raw_line in handle:
				line = raw_line.strip()
				if not line:
					continue
				parsed = ast.literal_eval(line)
				if not isinstance(parsed, (list, tuple)):
					raise ValueError(f"Expected a tuple-like record in {file_path.name}: {line[:80]}")

				flattened = self._flatten_record(parsed)
				if len(flattened) != len(self.header_columns):
					raise ValueError(
						f"Column count mismatch in {file_path.name}: "
						f"got {len(flattened)}, expected {len(self.header_columns)}"
					)
				rows.append(flattened)

		frame = pd.DataFrame(rows, columns=self.header_columns)
		frame.insert(0, "file_name", condition.file_name)
		frame.insert(1, "condition_key", condition.condition_key)
		frame.insert(2, "start_condition", condition.start_condition)
		frame.insert(3, "speed", condition.speed)
		frame.insert(4, "payload_lb", condition.payload_lb)
		frame.insert(5, "replicate", condition.replicate)
		frame.insert(6, "row_index", range(len(frame)))

		if "ROBOT_TIME" in frame.columns:
			frame = frame.sort_values("ROBOT_TIME", kind="stable").reset_index(drop=True)

		return frame

	def load_all(self) -> pd.DataFrame:
		"""Load all experiment files into a single data frame."""

		self.cartesian_error_data_frame = None

		frames = [self._parse_csv_file(path) for path in self._iter_csv_files()]
		if not frames:
			self.data_frame = pd.DataFrame(columns=[
				"file_name",
				"condition_key",
				"start_condition",
				"speed",
				"payload_lb",
				"replicate",
				"row_index",
				*self.header_columns,
			])
			return self.data_frame

		self.data_frame = pd.concat(frames, ignore_index=True)
		return self.data_frame

	def condition_summary(self) -> pd.DataFrame:
		"""Return a simple file-level summary for quick inspection."""

		if self.data_frame is None:
			self.load_all()

		assert self.data_frame is not None
		summary = (
			self.data_frame.groupby(
				["file_name", "condition_key", "start_condition", "speed", "payload_lb", "replicate"],
				as_index=False,
			)
			.agg(row_count=("row_index", "count"))
			.sort_values(["start_condition", "speed", "payload_lb", "replicate"])
			.reset_index(drop=True)
		)
		return summary

	def compute_joint_errors(self, frame: pd.DataFrame) -> pd.DataFrame:
		"""Return a copy of *frame* with six error columns added.

		Error columns are named `ERROR_J1` .. `ERROR_J6` and are computed as
		actual - target for each joint position.
		"""

		df = frame.copy()
		for j in range(1, 7):
			tcol = f"ROBOT_TARGET_JOINT_POSITIONS (J{j})"
			acol = f"ROBOT_ACTUAL_JOINT_POSITIONS (J{j})"
			ecol = f"ERROR_J{j}"
			if tcol in df.columns and acol in df.columns:
				df[ecol] = pd.to_numeric(df[acol], errors="coerce") - pd.to_numeric(df[tcol], errors="coerce")
			else:
				df[ecol] = float("nan")
		return df

	@classmethod
	def forward_kinematics(cls, joint_positions_deg: np.ndarray) -> np.ndarray:
		"""Return base-to-flange transforms for UR5 joint positions.

		The result has shape ``(n, 4, 4)``. This uses the nominal UR5 geometry,
		so it measures encoder-derived Cartesian tracking error rather than
		absolute physical TCP accuracy measured by an external instrument.
		"""

		joint_positions = np.asarray(joint_positions_deg, dtype=float)
		if joint_positions.ndim == 1:
			joint_positions = joint_positions.reshape(1, -1)
		if joint_positions.ndim != 2 or joint_positions.shape[1] != 6:
			raise ValueError("joint_positions_deg must have shape (n, 6)")

		theta = np.deg2rad(joint_positions)
		transforms = np.broadcast_to(np.eye(4), (len(theta), 4, 4)).copy()

		for joint in range(6):
			ct = np.cos(theta[:, joint])
			st = np.sin(theta[:, joint])
			ca = np.cos(cls.UR5_DH_ALPHA[joint])
			sa = np.sin(cls.UR5_DH_ALPHA[joint])
			a = cls.UR5_DH_A[joint]
			d = cls.UR5_DH_D[joint]

			joint_transform = np.zeros((len(theta), 4, 4), dtype=float)
			joint_transform[:, 0, 0] = ct
			joint_transform[:, 0, 1] = -st * ca
			joint_transform[:, 0, 2] = st * sa
			joint_transform[:, 0, 3] = a * ct
			joint_transform[:, 1, 0] = st
			joint_transform[:, 1, 1] = ct * ca
			joint_transform[:, 1, 2] = -ct * sa
			joint_transform[:, 1, 3] = a * st
			joint_transform[:, 2, 1] = sa
			joint_transform[:, 2, 2] = ca
			joint_transform[:, 2, 3] = d
			joint_transform[:, 3, 3] = 1.0

			transforms = transforms @ joint_transform

		return transforms

	def compute_cartesian_errors(self, frame: pd.DataFrame) -> pd.DataFrame:
		"""Add FK-derived Cartesian tracking-error columns to *frame*.

		Position components and magnitude are expressed in millimetres. A known
		TCP translation can be supplied through ``tcp_offset_m``; otherwise the
		UR5 flange origin is used. The
		orientation error is the shortest rotation angle in degrees between the
		target and actual flange orientations.
		"""

		df = frame.copy()
		target_cols = [f"ROBOT_TARGET_JOINT_POSITIONS (J{j})" for j in range(1, 7)]
		actual_cols = [f"ROBOT_ACTUAL_JOINT_POSITIONS (J{j})" for j in range(1, 7)]
		missing = [col for col in [*target_cols, *actual_cols] if col not in df.columns]
		if missing:
			raise ValueError(f"Missing joint-position columns required for FK: {missing}")

		target_joints = df[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
		actual_joints = df[actual_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
		valid = np.isfinite(target_joints).all(axis=1) & np.isfinite(actual_joints).all(axis=1)

		position_delta_mm = np.full((len(df), 3), np.nan)
		position_error_mm = np.full(len(df), np.nan)
		orientation_error_deg = np.full(len(df), np.nan)

		if valid.any():
			target_fk = self.forward_kinematics(target_joints[valid])
			actual_fk = self.forward_kinematics(actual_joints[valid])

			target_position = target_fk[:, :3, 3] + target_fk[:, :3, :3] @ self.tcp_offset_m
			actual_position = actual_fk[:, :3, 3] + actual_fk[:, :3, :3] @ self.tcp_offset_m
			delta = (actual_position - target_position) * 1000.0
			position_delta_mm[valid] = delta
			position_error_mm[valid] = np.linalg.norm(delta, axis=1)

			relative_rotation = np.swapaxes(target_fk[:, :3, :3], 1, 2) @ actual_fk[:, :3, :3]
			cos_angle = (np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0
			orientation_error_deg[valid] = np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

		df["FK_ERROR_X_MM"] = position_delta_mm[:, 0]
		df["FK_ERROR_Y_MM"] = position_delta_mm[:, 1]
		df["FK_ERROR_Z_MM"] = position_delta_mm[:, 2]
		df["FK_POSITION_ERROR_MM"] = position_error_mm
		df["FK_ORIENTATION_ERROR_DEG"] = orientation_error_deg
		return df

	def _joint_error_frame(self, relative_time: bool = True) -> tuple[pd.DataFrame, str]:
		"""Return loaded data with joint error columns and a plotting time axis."""

		df_all = self.data_frame if self.data_frame is not None else self.load_all()
		df = self.compute_joint_errors(df_all)

		sort_cols = ["file_name"]
		if "ROBOT_TIME" in df.columns:
			sort_cols.append("ROBOT_TIME")
		else:
			sort_cols.append("row_index")

		df = df.sort_values(sort_cols, kind="stable").copy()
		df["sample_index"] = df.groupby("file_name").cumcount()

		if "ROBOT_TIME" in df.columns:
			time = pd.to_numeric(df["ROBOT_TIME"], errors="coerce")
			if relative_time:
				time = time - time.groupby(df["file_name"]).transform("first")
				time_label = "Elapsed ROBOT_TIME"
			else:
				time_label = "ROBOT_TIME"

			df["plot_time"] = time
		else:
			df["plot_time"] = df["sample_index"]
			time_label = "Sample index"

		return df, time_label

	def _cartesian_error_frame(self, relative_time: bool = True) -> tuple[pd.DataFrame, str]:
		"""Return loaded data with FK Cartesian errors and plotting axes."""

		if self.cartesian_error_data_frame is None:
			df_all = self.data_frame if self.data_frame is not None else self.load_all()
			self.cartesian_error_data_frame = self.compute_cartesian_errors(df_all)
		df = self.cartesian_error_data_frame.copy()

		sort_cols = ["file_name", "ROBOT_TIME" if "ROBOT_TIME" in df.columns else "row_index"]
		df = df.sort_values(sort_cols, kind="stable").copy()
		df["sample_index"] = df.groupby("file_name").cumcount()
		file_sizes = df.groupby("file_name")["sample_index"].transform("max").clip(lower=1)
		df["progress"] = df["sample_index"] / file_sizes

		if "ROBOT_TIME" in df.columns:
			time = pd.to_numeric(df["ROBOT_TIME"], errors="coerce")
			if relative_time:
				time = time - time.groupby(df["file_name"]).transform("first")
				time_label = "Elapsed ROBOT_TIME (s)"
			else:
				time_label = "ROBOT_TIME (s)"
			df["plot_time"] = time
		else:
			df["plot_time"] = df["sample_index"]
			time_label = "Sample index"

		return df, time_label

	@staticmethod
	def _sanitize_filename(text: str) -> str:
		"""Make a safe file name from arbitrary condition text."""

		return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")

	@staticmethod
	def _format_level(factor: str, value: object) -> str:
		"""Format factor levels for legends."""

		if factor == "payload_lb":
			try:
				return f"{float(value):g} lb"
			except Exception:
				return str(value)

		return str(value)

	@staticmethod
	def _apply_condition_filter(
		df: pd.DataFrame,
		start_condition: Optional[str] = None,
		speed: Optional[str] = None,
		payload_lb: Optional[float] = None,
		replicate: Optional[int] = None,
	) -> pd.DataFrame:
		"""Filter by experiment conditions."""

		out = df

		if start_condition is not None:
			out = out[out["start_condition"] == start_condition]

		if speed is not None:
			out = out[out["speed"] == speed]

		if payload_lb is not None:
			out = out[pd.to_numeric(out["payload_lb"], errors="coerce") == float(payload_lb)]

		if replicate is not None:
			out = out[out["replicate"] == int(replicate)]

		return out

	def plot_joint_time_series(
		self,
		file_name: str,
		joints: Optional[Sequence[int]] = None,
		absolute: bool = False,
		relative_time: bool = True,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> Path:
		"""Plot joint error time-series for a single file and save the figure."""

		df_all = self.data_frame if self.data_frame is not None else self.load_all()
		df = df_all[df_all["file_name"] == file_name]

		if df.empty:
			raise ValueError(f"No data loaded for file: {file_name}")

		df = self.compute_joint_errors(df)

		if "ROBOT_TIME" in df.columns:
			df = df.sort_values("ROBOT_TIME", kind="stable").copy()
			time = pd.to_numeric(df["ROBOT_TIME"], errors="coerce")

			if relative_time:
				time = time - time.iloc[0]
				time_label = "Elapsed ROBOT_TIME"
			else:
				time_label = "ROBOT_TIME"
		else:
			time = pd.Series(range(len(df)), index=df.index)
			time_label = "Sample index"

		joints = list(joints) if joints is not None else list(range(1, 7))

		n = len(joints)
		cols = 3
		rows = ceil(n / cols)

		fig, axes = plt.subplots(
			rows,
			cols,
			figsize=(4.2 * cols, 2.9 * rows),
			sharex=True,
		)

		axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

		for idx, j in enumerate(joints):
			ax = axes[idx]
			ecol = f"ERROR_J{j}"

			if ecol not in df.columns:
				ax.text(0.5, 0.5, f"Joint {j} not available", ha="center", va="center")
				ax.set_title(f"J{j}")
				continue

			series = pd.to_numeric(df[ecol], errors="coerce")

			if absolute:
				series = series.abs()

			ax.plot(time, series, lw=0.7)
			ax.set_title(f"J{j} {'absolute ' if absolute else ''}error")
			ax.set_xlabel(time_label)
			ax.set_ylabel("|actual - target|" if absolute else "actual - target")
			ax.grid(alpha=0.3)

		for k in range(n, rows * cols):
			fig.delaxes(axes[k])

		fig.suptitle(f"Joint position error — {file_name}", fontsize=12)
		fig.tight_layout(rect=[0, 0, 1, 0.95])

		out_dir = Path(save_dir) if save_dir is not None else self.project_root / "output" / "plots" / "single_file"
		out_dir.mkdir(parents=True, exist_ok=True)

		safe_name = file_name.replace(".csv", "")
		suffix = "abs_error" if absolute else "signed_error"
		out_path = out_dir / f"{safe_name}_{suffix}.png"

		fig.savefig(out_path, dpi=dpi)
		plt.close(fig)

		return out_path

	def plot_all_files_joint_errors(
		self,
		joints: Optional[Sequence[int]] = None,
		absolute: bool = False,
		relative_time: bool = True,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> List[Path]:
		"""Create one joint error plot for every CSV file."""

		summary = self.condition_summary()
		saved: List[Path] = []

		for file_name in summary["file_name"]:
			out = self.plot_joint_time_series(
				file_name=file_name,
				joints=joints,
				absolute=absolute,
				relative_time=relative_time,
				save_dir=save_dir,
				dpi=dpi,
			)
			saved.append(out)

		return saved

	def plot_cartesian_time_series(
		self,
		file_name: str,
		relative_time: bool = True,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> Path:
		"""Plot FK-derived Cartesian tracking error for one experiment file."""

		df, time_label = self._cartesian_error_frame(relative_time=relative_time)
		df = df[df["file_name"] == file_name]
		if df.empty:
			raise ValueError(f"No data loaded for file: {file_name}")

		metrics = [
			("FK_ERROR_X_MM", "X error", "actual - target (mm)"),
			("FK_ERROR_Y_MM", "Y error", "actual - target (mm)"),
			("FK_ERROR_Z_MM", "Z error", "actual - target (mm)"),
			("FK_POSITION_ERROR_MM", "3D position error", "error magnitude (mm)"),
			("FK_ORIENTATION_ERROR_DEG", "Orientation error", "rotation error (deg)"),
		]
		fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.2), sharex=True)
		axes = axes.flatten()

		for ax, (column, title, ylabel) in zip(axes, metrics):
			ax.plot(df["plot_time"], df[column], lw=0.7)
			ax.set_title(title)
			ax.set_xlabel(time_label)
			ax.set_ylabel(ylabel)
			ax.grid(alpha=0.3)

		fig.delaxes(axes[-1])
		fig.suptitle(
			f"UR5 FK-derived Cartesian tracking error\n{file_name}",
			fontsize=12,
		)
		fig.tight_layout(rect=[0, 0, 1, 0.93])

		out_dir = Path(save_dir) if save_dir is not None else self.project_root / "output" / "plots" / "cartesian_single_file"
		out_dir.mkdir(parents=True, exist_ok=True)
		out_path = out_dir / f"{file_name.replace('.csv', '')}_cartesian_error.png"
		fig.savefig(out_path, dpi=dpi)
		plt.close(fig)
		return out_path

	def plot_all_files_cartesian_errors(
		self,
		relative_time: bool = True,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> List[Path]:
		"""Create one FK Cartesian-error plot for every CSV file."""

		return [
			self.plot_cartesian_time_series(
				file_name=file_name,
				relative_time=relative_time,
				save_dir=save_dir,
				dpi=dpi,
			)
			for file_name in self.condition_summary()["file_name"]
		]

	def plot_cartesian_factor_effect(
		self,
		factor: str,
		start_condition: Optional[str] = None,
		speed: Optional[str] = None,
		payload_lb: Optional[float] = None,
		time_bin_seconds: float = 1.0,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> Path:
		"""Compare FK Cartesian errors over elapsed robot time.

		Each run is reduced to a median within fixed-width time bins. The plotted
		line and band are the median and IQR across replicate runs, respectively.
		"""

		valid_factors = {"start_condition", "speed", "payload_lb"}
		if factor not in valid_factors:
			raise ValueError(f"`factor` must be one of {sorted(valid_factors)}, got: {factor}")
		fixed_values = {
			"start_condition": start_condition,
			"speed": speed,
			"payload_lb": payload_lb,
		}
		if fixed_values[factor] is not None:
			raise ValueError(f"Do not fix `{factor}` when it is the compared factor.")

		df, _ = self._cartesian_error_frame(relative_time=True)
		df = self._apply_condition_filter(
			df,
			start_condition=start_condition if factor != "start_condition" else None,
			speed=speed if factor != "speed" else None,
			payload_lb=payload_lb if factor != "payload_lb" else None,
		)
		if df.empty or df[factor].nunique(dropna=True) <= 1:
			raise ValueError("At least two factor levels are required for comparison.")

		non_factor_cols = [col for col in valid_factors if col != factor]
		varying = [col for col in non_factor_cols if df[col].nunique(dropna=True) > 1]
		if varying:
			raise ValueError(f"All non-factor conditions must be fixed. Currently varying: {varying}")
		if time_bin_seconds <= 0:
			raise ValueError("time_bin_seconds must be greater than zero.")

		df = df.copy()
		df = df[np.isfinite(df["plot_time"])].copy()
		df["time_bin"] = np.floor(df["plot_time"] / time_bin_seconds).astype(int)
		metrics = [
			("FK_POSITION_ERROR_MM", "3D position tracking error", "Position error (mm)"),
			("FK_ORIENTATION_ERROR_DEG", "Orientation tracking error", "Orientation error (deg)"),
		]
		fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)

		for ax, (metric, title, ylabel) in zip(axes, metrics):
			per_run = (
				df.groupby([factor, "file_name", "time_bin"], as_index=False)
				.agg(value=(metric, "median"))
			)
			summary = (
				per_run.groupby([factor, "time_bin"], as_index=False)
				.agg(
					median=("value", "median"),
					lower=("value", lambda values: values.quantile(0.25)),
					upper=("value", lambda values: values.quantile(0.75)),
					replicate_count=("value", "count"),
				)
			)

			for level, group in summary.groupby(factor, sort=True):
				x = (group["time_bin"] + 0.5) * time_bin_seconds
				label = self._format_level(factor, level)
				line = ax.plot(x, group["median"], lw=1.5, label=f"{label} median")[0]
				ax.fill_between(
					x,
					group["lower"],
					group["upper"],
					color=line.get_color(),
					alpha=0.28,
					linewidth=0,
					label=f"{label} IQR",
				)
				ax.plot(x, group["lower"], color=line.get_color(), lw=0.7, ls="--", alpha=0.8)
				ax.plot(x, group["upper"], color=line.get_color(), lw=0.7, ls="--", alpha=0.8)

			ax.set_title(title)
			ax.set_xlabel("Elapsed ROBOT_TIME (s)")
			ax.set_ylabel(ylabel)
			ax.grid(alpha=0.3)

		axes[0].legend(fontsize=8, ncol=2)
		fixed_text = ", ".join(
			f"{key}={self._format_level(key, value)}"
			for key, value in fixed_values.items()
			if value is not None
		)
		fig.suptitle(
			f"Effect of {factor} on FK-derived Cartesian tracking error\n"
			f"{fixed_text} | {time_bin_seconds:g} s bins, median and replicate IQR",
			fontsize=12,
		)
		fig.tight_layout(rect=[0, 0, 1, 0.9])

		out_dir = Path(save_dir) if save_dir is not None else self.project_root / "output" / "plots" / "cartesian_factor_effects"
		out_dir.mkdir(parents=True, exist_ok=True)
		name_parts = [
			f"cartesian_effect_{factor}",
			f"start_{start_condition}" if start_condition is not None else None,
			f"speed_{speed}" if speed is not None else None,
			f"payload_{payload_lb:g}lb" if payload_lb is not None else None,
		]
		file_stem = self._sanitize_filename("_".join(str(part) for part in name_parts if part))
		out_path = out_dir / f"{file_stem}.png"
		fig.savefig(out_path, dpi=dpi)
		plt.close(fig)
		return out_path

	def plot_all_cartesian_factor_effects(
		self,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> List[Path]:
		"""Generate every available FK Cartesian factor comparison."""

		df, _ = self._cartesian_error_frame(relative_time=True)
		starts = sorted(df["start_condition"].dropna().unique())
		speeds = sorted(df["speed"].dropna().unique())
		payloads = sorted(df["payload_lb"].dropna().unique(), key=float)
		saved: List[Path] = []

		for speed_value in speeds:
			for payload_value in payloads:
				subset = self._apply_condition_filter(df, speed=speed_value, payload_lb=float(payload_value))
				if subset["start_condition"].nunique(dropna=True) > 1:
					saved.append(self.plot_cartesian_factor_effect(
						"start_condition", speed=speed_value, payload_lb=float(payload_value),
						save_dir=save_dir, dpi=dpi,
					))

		for start_value in starts:
			for payload_value in payloads:
				subset = self._apply_condition_filter(df, start_condition=start_value, payload_lb=float(payload_value))
				if subset["speed"].nunique(dropna=True) > 1:
					saved.append(self.plot_cartesian_factor_effect(
						"speed", start_condition=start_value, payload_lb=float(payload_value),
						save_dir=save_dir, dpi=dpi,
					))

		for start_value in starts:
			for speed_value in speeds:
				subset = self._apply_condition_filter(df, start_condition=start_value, speed=speed_value)
				if subset["payload_lb"].nunique(dropna=True) > 1:
					saved.append(self.plot_cartesian_factor_effect(
						"payload_lb", start_condition=start_value, speed=speed_value,
						save_dir=save_dir, dpi=dpi,
					))

		return saved

	def plot_factor_effect_time_series(
		self,
		factor: str,
		start_condition: Optional[str] = None,
		speed: Optional[str] = None,
		payload_lb: Optional[float] = None,
		replicate: Optional[int] = None,
		joints: Optional[Sequence[int]] = None,
		absolute: bool = True,
		aggregate_replicates: bool = True,
		relative_time: bool = True,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> Path:
		"""Plot the effect of one experimental factor on joint error.

		`factor` should be one of:
		- "start_condition"
		- "speed"
		- "payload_lb"

		Example:
			plot_factor_effect_time_series(
				factor="payload_lb",
				start_condition="normalstart",
				speed="fullspeed",
			)

		This compares payload levels while start_condition and speed are fixed.
		"""

		valid_factors = {"start_condition", "speed", "payload_lb"}

		if factor not in valid_factors:
			raise ValueError(f"`factor` must be one of {sorted(valid_factors)}, got: {factor}")

		fixed_values = {
			"start_condition": start_condition,
			"speed": speed,
			"payload_lb": payload_lb,
		}

		if fixed_values[factor] is not None:
			raise ValueError(
				f"Do not fix `{factor}` when it is the factor being compared."
			)

		df, time_label = self._joint_error_frame(relative_time=relative_time)

		df = self._apply_condition_filter(
			df,
			start_condition=start_condition if factor != "start_condition" else None,
			speed=speed if factor != "speed" else None,
			payload_lb=payload_lb if factor != "payload_lb" else None,
			replicate=replicate,
		)

		if df.empty:
			raise ValueError("No data found for the selected condition filter.")

		joints = list(joints) if joints is not None else list(range(1, 7))

		# Averaging is meaningful only when all non-factor variables are fixed.
		if aggregate_replicates:
			non_factor_cols = [col for col in ["start_condition", "speed", "payload_lb"] if col != factor]
			varying_cols = [col for col in non_factor_cols if df[col].nunique(dropna=True) > 1]

			if varying_cols:
				raise ValueError(
					"aggregate_replicates=True requires all non-factor variables to be fixed. "
					f"Currently varying: {varying_cols}"
				)

		n = len(joints)
		cols = 3
		rows = ceil(n / cols)

		fig, axes = plt.subplots(
			rows,
			cols,
			figsize=(4.4 * cols, 3.0 * rows),
			sharex=True,
		)

		axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

		for idx, j in enumerate(joints):
			ax = axes[idx]
			ecol = f"ERROR_J{j}"

			if ecol not in df.columns:
				ax.text(0.5, 0.5, f"Joint {j} not available", ha="center", va="center")
				ax.set_title(f"J{j}")
				continue

			plot_df = df.copy()
			plot_df[ecol] = pd.to_numeric(plot_df[ecol], errors="coerce")

			if absolute:
				plot_df[ecol] = plot_df[ecol].abs()

			if aggregate_replicates:
				agg = (
					plot_df
					.groupby([factor, "sample_index"], as_index=False)
					.agg(
						plot_time=("plot_time", "mean"),
						error_value=(ecol, "mean"),
					)
				)

				for level, group in agg.groupby(factor, sort=True):
					label = self._format_level(factor, level)
					ax.plot(
						group["plot_time"],
						group["error_value"],
						lw=1.2,
						label=label,
					)

			else:
				for file_name, group in plot_df.groupby("file_name", sort=False):
					meta = group.iloc[0]
					level = meta[factor]
					label = (
						f"{self._format_level(factor, level)} | "
						f"rep {int(meta['replicate'])}"
					)

					ax.plot(
						group["plot_time"],
						group[ecol],
						lw=0.6,
						alpha=0.8,
						label=label,
					)

			ax.set_title(f"J{j}")
			ax.set_xlabel(time_label)
			ax.set_ylabel("|actual - target|" if absolute else "actual - target")
			ax.grid(alpha=0.3)

		for k in range(n, rows * cols):
			fig.delaxes(axes[k])

		handles, labels = axes[0].get_legend_handles_labels()
		unique = dict(zip(labels, handles))

		if unique:
			fig.legend(
				unique.values(),
				unique.keys(),
				loc="lower center",
				ncol=min(4, len(unique)),
				fontsize=9,
			)

		factor_label = {
			"start_condition": "start condition / temperature",
			"speed": "speed",
			"payload_lb": "payload",
		}[factor]

		fixed_text_parts = []
		if start_condition is not None:
			fixed_text_parts.append(f"start={start_condition}")
		if speed is not None:
			fixed_text_parts.append(f"speed={speed}")
		if payload_lb is not None:
			fixed_text_parts.append(f"payload={payload_lb:g} lb")
		if replicate is not None:
			fixed_text_parts.append(f"replicate={replicate}")

		fixed_text = ", ".join(fixed_text_parts) if fixed_text_parts else "no fixed condition"
		metric_text = "absolute joint error" if absolute else "signed joint error"
		agg_text = "mean over replicates" if aggregate_replicates else "raw replicate traces"

		fig.suptitle(
			f"Effect of {factor_label} on {metric_text}\n"
			f"{fixed_text} | {agg_text}",
			fontsize=12,
		)

		fig.tight_layout(rect=[0, 0.08, 1, 0.92])

		out_dir = Path(save_dir) if save_dir is not None else self.project_root / "output" / "plots" / "factor_effects"
		out_dir.mkdir(parents=True, exist_ok=True)

		name_parts = [
			f"effect_{factor}",
			f"start_{start_condition}" if start_condition is not None else None,
			f"speed_{speed}" if speed is not None else None,
			f"payload_{payload_lb:g}lb" if payload_lb is not None else None,
			f"rep_{replicate}" if replicate is not None else None,
			"abs" if absolute else "signed",
			"mean" if aggregate_replicates else "raw",
		]

		file_stem = self._sanitize_filename("_".join(str(p) for p in name_parts if p is not None))
		out_path = out_dir / f"{file_stem}.png"

		fig.savefig(out_path, dpi=dpi)
		plt.close(fig)

		return out_path

	def plot_all_factor_effects(
		self,
		joints: Optional[Sequence[int]] = None,
		absolute: bool = True,
		aggregate_replicates: bool = True,
		relative_time: bool = True,
		save_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> List[Path]:
		"""Automatically generate factor-effect plots.

		This creates:
		1. start_condition effect plots while speed and payload are fixed
		2. speed effect plots while start_condition and payload are fixed
		3. payload effect plots while start_condition and speed are fixed
		"""

		df, _ = self._joint_error_frame(relative_time=relative_time)

		start_conditions = sorted(df["start_condition"].dropna().unique())
		speeds = sorted(df["speed"].dropna().unique())
		payloads = sorted(df["payload_lb"].dropna().unique(), key=float)

		saved: List[Path] = []

		# 1. Temperature / start condition effect
		for speed_value in speeds:
			for payload_value in payloads:
				subset = self._apply_condition_filter(
					df,
					speed=speed_value,
					payload_lb=float(payload_value),
				)

				if subset["start_condition"].nunique(dropna=True) <= 1:
					continue

				out = self.plot_factor_effect_time_series(
					factor="start_condition",
					speed=speed_value,
					payload_lb=float(payload_value),
					joints=joints,
					absolute=absolute,
					aggregate_replicates=aggregate_replicates,
					relative_time=relative_time,
					save_dir=save_dir,
					dpi=dpi,
				)
				saved.append(out)

		# 2. Speed effect
		for start_value in start_conditions:
			for payload_value in payloads:
				subset = self._apply_condition_filter(
					df,
					start_condition=start_value,
					payload_lb=float(payload_value),
				)

				if subset["speed"].nunique(dropna=True) <= 1:
					continue

				out = self.plot_factor_effect_time_series(
					factor="speed",
					start_condition=start_value,
					payload_lb=float(payload_value),
					joints=joints,
					absolute=absolute,
					aggregate_replicates=aggregate_replicates,
					relative_time=relative_time,
					save_dir=save_dir,
					dpi=dpi,
				)
				saved.append(out)

		# 3. Payload effect
		for start_value in start_conditions:
			for speed_value in speeds:
				subset = self._apply_condition_filter(
					df,
					start_condition=start_value,
					speed=speed_value,
				)

				if subset["payload_lb"].nunique(dropna=True) <= 1:
					continue

				out = self.plot_factor_effect_time_series(
					factor="payload_lb",
					start_condition=start_value,
					speed=speed_value,
					joints=joints,
					absolute=absolute,
					aggregate_replicates=aggregate_replicates,
					relative_time=relative_time,
					save_dir=save_dir,
					dpi=dpi,
				)
				saved.append(out)

		return saved


def main() -> None:
	analyzer = UR5DegradationAnalyzer()
	data_frame = analyzer.load_all()

	print(f"Loaded rows: {len(data_frame):,}")
	print(f"Loaded columns: {len(data_frame.columns):,}")
	print(analyzer.condition_summary().to_string(index=False))

	# 1. 모든 개별 파일에 대해 joint error time-series 저장
	single_paths = analyzer.plot_all_files_joint_errors(
		absolute=False,
		save_dir=analyzer.project_root / "output" / "plots" / "single_file_signed",
	)

	# 2. 모든 개별 파일에 대해 absolute joint error time-series 저장
	single_abs_paths = analyzer.plot_all_files_joint_errors(
		absolute=True,
		save_dir=analyzer.project_root / "output" / "plots" / "single_file_abs",
	)

	# 3. start_condition, speed, payload 각각의 영향 비교 plot 저장
	factor_paths = analyzer.plot_all_factor_effects(
		absolute=True,
		aggregate_replicates=True,
		save_dir=analyzer.project_root / "output" / "plots" / "factor_effects",
	)

	# 4. FK-derived Cartesian tracking error for each experiment file
	cartesian_paths = analyzer.plot_all_files_cartesian_errors(
		save_dir=analyzer.project_root / "output" / "plots" / "cartesian_single_file",
	)

	# 5. Cartesian factor comparisons aligned by normalized run progress
	cartesian_factor_paths = analyzer.plot_all_cartesian_factor_effects(
		save_dir=analyzer.project_root / "output" / "plots" / "cartesian_factor_effects",
	)

	print(f"Saved single-file signed plots: {len(single_paths)}")
	print(f"Saved single-file absolute plots: {len(single_abs_paths)}")
	print(f"Saved factor-effect plots: {len(factor_paths)}")
	print(f"Saved Cartesian single-file plots: {len(cartesian_paths)}")
	print(f"Saved Cartesian factor-effect plots: {len(cartesian_factor_paths)}")


if __name__ == "__main__":
	main()
