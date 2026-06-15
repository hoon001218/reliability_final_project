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
from typing import Dict, List, Optional, Sequence

import pandas as pd
import numpy as np
from scipy import stats
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


@dataclass(frozen=True)
class DistributionFit:
	"""MLE fit and probability-plot diagnostics for one error sample."""

	metric: str
	speed_scale: float
	distribution: str
	parameters: tuple[float, ...]
	log_likelihood: float
	aic: float
	probability_plot_r2: float
	probability_plot_slope: float
	probability_plot_intercept: float
	sample_count: int


class UR5DegradationAnalyzer:
	"""Load and organize the UR5 accelerated-aging experiment results."""

	# Nominal standard-DH dimensions for the original UR5 (CB-series), in metres.
	# The raw controller joint positions are recorded in degrees.
	UR5_DH_A = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
	UR5_DH_D = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
	UR5_DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
	SPEED_SCALE = {"halfspeed": 0.5, "fullspeed": 1.0}
	RELIABILITY_METRICS = {
		"position": ("FK_POSITION_ERROR_MM", "Position error (mm)"),
		"orientation": ("FK_ORIENTATION_ERROR_DEG", "Orientation error (deg)"),
	}
	DISTRIBUTIONS = ("normal", "lognormal", "weibull")

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

	@staticmethod
	def _distribution_parameter_names(distribution: str) -> tuple[str, ...]:
		if distribution == "normal":
			return ("mean", "std")
		if distribution == "lognormal":
			return ("shape_sigma", "scale_exp_mu")
		if distribution == "weibull":
			return ("shape", "scale")
		raise ValueError(f"Unsupported distribution: {distribution}")

	@staticmethod
	def _fit_distribution_values(values: np.ndarray, distribution: str) -> tuple[float, ...]:
		"""Fit one candidate distribution by maximum likelihood."""

		values = np.asarray(values, dtype=float)
		values = values[np.isfinite(values)]
		if len(values) < 2:
			raise ValueError("At least two finite error values are required.")

		if distribution == "normal":
			location, scale = stats.norm.fit(values)
			return float(location), float(scale)

		positive_values = np.maximum(values, np.finfo(float).tiny)
		if distribution == "lognormal":
			shape, _, scale = stats.lognorm.fit(positive_values, floc=0.0)
			return float(shape), float(scale)
		if distribution == "weibull":
			shape, _, scale = stats.weibull_min.fit(positive_values, floc=0.0)
			return float(shape), float(scale)

		raise ValueError(f"Unsupported distribution: {distribution}")

	@staticmethod
	def _distribution_object(distribution: str, parameters: Sequence[float]):
		"""Return a frozen SciPy distribution for the stored parameterization."""

		if distribution == "normal":
			return stats.norm(loc=parameters[0], scale=parameters[1])
		if distribution == "lognormal":
			return stats.lognorm(s=parameters[0], loc=0.0, scale=parameters[1])
		if distribution == "weibull":
			return stats.weibull_min(c=parameters[0], loc=0.0, scale=parameters[1])
		raise ValueError(f"Unsupported distribution: {distribution}")

	@staticmethod
	def _probability_plot_points(
		values: np.ndarray,
		distribution: str,
		max_points: int = 2000,
		log_error_axis: bool = False,
	) -> tuple[np.ndarray, np.ndarray, float, float, float]:
		"""Return probability-paper coordinates and their linear regression."""

		observed_full = np.sort(np.asarray(values, dtype=float))
		observed_full = observed_full[np.isfinite(observed_full)]
		plotting_positions = (
			np.arange(1, len(observed_full) + 1) - 0.5
		) / len(observed_full)

		if distribution == "normal":
			probability_scale = stats.norm.ppf(plotting_positions)
			observed_transformed = (
				np.log(np.maximum(observed_full, np.finfo(float).tiny))
				if log_error_axis
				else observed_full
			)
		elif distribution == "lognormal":
			probability_scale = stats.norm.ppf(plotting_positions)
			observed_transformed = np.log(
				np.maximum(observed_full, np.finfo(float).tiny)
			)
		elif distribution == "weibull":
			probability_scale = np.log(-np.log1p(-plotting_positions))
			observed_transformed = np.log(
				np.maximum(observed_full, np.finfo(float).tiny)
			)
		else:
			raise ValueError(f"Unsupported distribution: {distribution}")

		valid = np.isfinite(probability_scale) & np.isfinite(observed_transformed)
		probability_scale = probability_scale[valid]
		observed_transformed = observed_transformed[valid]

		regression = stats.linregress(observed_transformed, probability_scale)
		if len(observed_transformed) > max_points:
			indices = np.linspace(
				0,
				len(observed_transformed) - 1,
				max_points,
			).round().astype(int)
		else:
			indices = np.arange(len(observed_transformed))

		return (
			observed_transformed[indices],
			probability_scale[indices],
			float(regression.rvalue ** 2),
			float(regression.slope),
			float(regression.intercept),
		)

	def _normal_start_reliability_frame(self) -> pd.DataFrame:
		"""Return normal-start FK errors with numeric speed-scale metadata."""

		df, _ = self._cartesian_error_frame(relative_time=True)
		df = df[df["start_condition"] == "normalstart"].copy()
		df["speed_scale"] = df["speed"].map(self.SPEED_SCALE)
		if df["speed_scale"].isna().any():
			raise ValueError("An unknown speed label was found in the normal-start data.")
		return df

	def fit_reliability_distributions(
		self,
		probability_plot_dir: Optional[Path | str] = None,
		distribution_plot_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> tuple[List[DistributionFit], str]:
		"""Fit candidates and select one common family using probability plots.

		Payload levels are pooled. A separate MLE fit is made for each combination
		of error metric and measured speed. The selected family maximizes the mean
		probability-plot R-squared across the four datasets; mean AIC rank is used
		only as a tie-breaker.
		"""

		df = self._normal_start_reliability_frame()
		prob_dir = Path(probability_plot_dir) if probability_plot_dir is not None else self.project_root / "output" / "reliability" / "probability_plots"
		dist_dir = Path(distribution_plot_dir) if distribution_plot_dir is not None else self.project_root / "output" / "reliability" / "distribution_fits"
		prob_dir.mkdir(parents=True, exist_ok=True)
		dist_dir.mkdir(parents=True, exist_ok=True)

		fits: List[DistributionFit] = []
		for metric_name, (column, axis_label) in self.RELIABILITY_METRICS.items():
			for speed_scale, group in df.groupby("speed_scale", sort=True):
				values = pd.to_numeric(group[column], errors="coerce").to_numpy()
				values = values[np.isfinite(values)]
				figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
				combined_figure, combined_axis = plt.subplots(figsize=(7.2, 4.8))
				x_upper = np.quantile(values, 0.9995)
				x_grid = np.linspace(0.0, max(x_upper, np.finfo(float).eps), 800)
				positive_values = np.maximum(values, np.finfo(float).tiny)
				log_values = np.sort(np.log(positive_values))
				plotting_positions = (
					np.arange(1, len(log_values) + 1) - 0.5
				) / len(log_values)
				if len(log_values) > 2000:
					combined_indices = np.linspace(
						0, len(log_values) - 1, 2000,
					).round().astype(int)
				else:
					combined_indices = np.arange(len(log_values))
				combined_axis.scatter(
					log_values[combined_indices],
					plotting_positions[combined_indices],
					s=7,
					alpha=0.3,
					color="0.35",
					label="Empirical",
				)
				combined_log_grid = np.linspace(log_values.min(), log_values.max(), 800)
				combined_error_grid = np.exp(combined_log_grid)

				fit_figure, fit_axis = plt.subplots(figsize=(7.0, 4.5))
				fit_axis.hist(values, bins=100, density=True, alpha=0.32, color="0.55", label="Observed")

				for axis, distribution in zip(axes, self.DISTRIBUTIONS):
					parameters = self._fit_distribution_values(values, distribution)
					distribution_object = self._distribution_object(distribution, parameters)
					(
						_,
						_,
						probability_r2,
						probability_slope,
						probability_intercept,
					) = self._probability_plot_points(
						values,
						distribution,
					)
					(
						observed,
						probability_coordinates,
						plot_probability_r2,
						plot_probability_slope,
						plot_probability_intercept,
					) = self._probability_plot_points(
						values,
						distribution,
						log_error_axis=True,
					)
					likelihood_values = (
						values
						if distribution == "normal"
						else np.maximum(values, np.finfo(float).tiny)
					)
					log_likelihood = float(np.sum(distribution_object.logpdf(likelihood_values)))
					aic = 2 * len(parameters) - 2 * log_likelihood
					fits.append(DistributionFit(
						metric=metric_name,
						speed_scale=float(speed_scale),
						distribution=distribution,
						parameters=parameters,
						log_likelihood=log_likelihood,
						aic=float(aic),
						probability_plot_r2=probability_r2,
						probability_plot_slope=probability_slope,
						probability_plot_intercept=probability_intercept,
						sample_count=len(values),
					))

					axis.scatter(observed, probability_coordinates, s=6, alpha=0.4)
					line_x = np.array([observed.min(), observed.max()])
					axis.plot(
						line_x,
						plot_probability_intercept + plot_probability_slope * line_x,
						color="tab:red",
						lw=1.4,
						# label="Linear fit",
					)
					axis.set_title(
						f"{distribution.title()}\n"
						# f"y={plot_probability_intercept:.3g}+{plot_probability_slope:.3g}x"
					)
					axis.set_xlabel("log(observed error)")
					probability_ticks = np.array([
						0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999,
					])
					if distribution in {"normal", "lognormal"}:
						tick_locations = stats.norm.ppf(probability_ticks)
					else:
						tick_locations = np.log(-np.log1p(-probability_ticks))
					y_min, y_max = probability_coordinates.min(), probability_coordinates.max()
					tick_mask = (tick_locations >= y_min) & (tick_locations <= y_max)
					axis.set_yticks(
						tick_locations[tick_mask],
						[f"{probability:g}" for probability in probability_ticks[tick_mask]],
					)
					axis.tick_params(axis="y", labelsize=8)
					axis.set_ylabel("Cumulative probability F(error)")
					axis.grid(alpha=0.25)
					axis.legend(fontsize=7)

					fit_axis.plot(x_grid, distribution_object.pdf(x_grid), lw=1.4, label=distribution.title())
					combined_axis.plot(
						combined_log_grid,
						distribution_object.cdf(combined_error_grid),
						lw=1.6,
						label=f"{distribution.title()} fit",
					)

				figure.suptitle(
					f"Probability plots: {metric_name} error, speed scale={speed_scale:g}\n"
					"Normal-start data; payload levels pooled",
					fontsize=12,
				)
				figure.tight_layout(rect=[0, 0, 1, 0.9])
				figure.savefig(prob_dir / f"{metric_name}_speed_{speed_scale:g}_probability_plots.png", dpi=dpi)
				plt.close(figure)

				combined_axis.set_xlabel("log(error)")
				combined_axis.set_ylabel("Cumulative probability F(error)")
				combined_axis.set_ylim(0.0, 1.0)
				combined_axis.set_title(
					f"Combined probability plot: {metric_name}, speed scale={speed_scale:g}\n"
					"Normal-start data; payload levels pooled"
				)
				combined_axis.legend(fontsize=8)
				combined_axis.grid(alpha=0.25)
				combined_figure.tight_layout()
				combined_figure.savefig(
					prob_dir / f"{metric_name}_speed_{speed_scale:g}_combined_probability_plot.png",
					dpi=dpi,
				)
				plt.close(combined_figure)

				fit_axis.set_xlim(0.0, x_upper)
				fit_axis.set_xlabel(axis_label)
				fit_axis.set_ylabel("Probability density")
				fit_axis.set_title(f"MLE distribution fits: {metric_name}, speed scale={speed_scale:g}")
				fit_axis.legend()
				fit_axis.grid(alpha=0.25)
				fit_figure.tight_layout()
				fit_figure.savefig(dist_dir / f"{metric_name}_speed_{speed_scale:g}_distribution_fits.png", dpi=dpi)
				plt.close(fit_figure)

		fit_table = pd.DataFrame([
			{
				"metric": fit.metric,
				"speed_scale": fit.speed_scale,
				"distribution": fit.distribution,
				"parameters": ", ".join(f"{value:.12g}" for value in fit.parameters),
				"log_likelihood": fit.log_likelihood,
				"aic": fit.aic,
				"probability_plot_r2": fit.probability_plot_r2,
				"probability_plot_slope": fit.probability_plot_slope,
				"probability_plot_intercept": fit.probability_plot_intercept,
				"sample_count": fit.sample_count,
			}
			for fit in fits
		])
		fit_table["aic_rank_within_dataset"] = fit_table.groupby(
			["metric", "speed_scale"]
		)["aic"].rank(method="average")
		selection = (
			fit_table.groupby("distribution", as_index=False)
			.agg(
				mean_probability_plot_r2=("probability_plot_r2", "mean"),
				minimum_probability_plot_r2=("probability_plot_r2", "min"),
				mean_aic_rank=("aic_rank_within_dataset", "mean"),
			)
			.sort_values(
				["mean_probability_plot_r2", "mean_aic_rank"],
				ascending=[False, True],
			)
			.reset_index(drop=True)
		)
		selected_distribution = str(selection.iloc[0]["distribution"])
		selection["selected"] = selection["distribution"] == selected_distribution

		output_dir = prob_dir.parent
		fit_table.to_csv(output_dir / "distribution_fit_details.csv", index=False)
		selection.to_csv(output_dir / "model_selection_summary.csv", index=False)
		return fits, selected_distribution

	@staticmethod
	def _fit_lookup(
		fits: Sequence[DistributionFit],
		metric: str,
		speed_scale: float,
		distribution: str,
	) -> DistributionFit:
		for fit in fits:
			if (
				fit.metric == metric
				and np.isclose(fit.speed_scale, speed_scale)
				and fit.distribution == distribution
			):
				return fit
		raise ValueError(f"No fit found for {metric}, speed={speed_scale}, {distribution}")

	def interpolate_distribution_parameters(
		self,
		fits: Sequence[DistributionFit],
		selected_distribution: str,
		metric: str,
		speed_scale: float,
	) -> np.ndarray:
		"""Linearly interpolate selected MLE parameters over speed 0.5 to 1.0."""

		if not 0.5 <= speed_scale <= 1.0:
			raise ValueError("speed_scale interpolation is limited to [0.5, 1.0].")
		low = self._fit_lookup(fits, metric, 0.5, selected_distribution)
		high = self._fit_lookup(fits, metric, 1.0, selected_distribution)
		weight = (speed_scale - 0.5) / 0.5
		return np.asarray(low.parameters) + weight * (
			np.asarray(high.parameters) - np.asarray(low.parameters)
		)

	def error_reliability(
		self,
		fits: Sequence[DistributionFit],
		selected_distribution: str,
		metric: str,
		speed_scale: float,
		tolerance: float,
	) -> float:
		"""Return P(error <= tolerance) from the interpolated distribution."""

		if tolerance <= 0:
			raise ValueError("tolerance must be greater than zero.")
		parameters = self.interpolate_distribution_parameters(
			fits,
			selected_distribution,
			metric,
			speed_scale,
		)
		return float(self._distribution_object(selected_distribution, parameters).cdf(tolerance))

	def reliability_over_speed(
		self,
		fits: Sequence[DistributionFit],
		selected_distribution: str,
		metric: str,
		speeds: np.ndarray,
		tolerance: float,
	) -> np.ndarray:
		"""Vectorized reliability evaluation over speeds in the interpolation range."""

		speeds = np.asarray(speeds, dtype=float)
		if np.any((speeds < 0.5) | (speeds > 1.0)):
			raise ValueError("All speed values must be within [0.5, 1.0].")
		low = np.asarray(self._fit_lookup(fits, metric, 0.5, selected_distribution).parameters)
		high = np.asarray(self._fit_lookup(fits, metric, 1.0, selected_distribution).parameters)
		weights = ((speeds - 0.5) / 0.5)[:, None]
		parameters = low + weights * (high - low)

		if selected_distribution == "normal":
			return stats.norm.cdf(tolerance, loc=parameters[:, 0], scale=parameters[:, 1])
		if selected_distribution == "lognormal":
			return stats.lognorm.cdf(tolerance, s=parameters[:, 0], loc=0.0, scale=parameters[:, 1])
		if selected_distribution == "weibull":
			return stats.weibull_min.cdf(tolerance, c=parameters[:, 0], loc=0.0, scale=parameters[:, 1])
		raise ValueError(f"Unsupported distribution: {selected_distribution}")

	def maximum_allowable_speed(
		self,
		fits: Sequence[DistributionFit],
		selected_distribution: str,
		metric: str,
		tolerance: float,
		required_reliability: float,
		grid_size: int = 5001,
	) -> Dict[str, object]:
		"""Find the largest speed satisfying one error-reliability constraint."""

		if not 0 < required_reliability < 1:
			raise ValueError("required_reliability must be between zero and one.")
		if metric not in self.RELIABILITY_METRICS:
			raise ValueError(f"Unknown metric: {metric}")

		speeds = np.linspace(0.5, 1.0, grid_size)
		reliability = self.reliability_over_speed(
			fits, selected_distribution, metric, speeds, tolerance
		)
		feasible = reliability >= required_reliability

		if feasible.any():
			index = int(np.flatnonzero(feasible)[-1])
			status = "feasible"
			maximum_speed = float(speeds[index])
		else:
			index = 0
			status = "not_feasible_at_0.5"
			maximum_speed = float("nan")

		return {
			"metric": metric,
			"tolerance": tolerance,
			"tolerance_unit": "mm" if metric == "position" else "deg",
			"required_reliability": required_reliability,
			"maximum_speed_scale": maximum_speed,
			"reliability_at_limit": float(reliability[index]),
			"status": status,
		}

	def plot_selected_parameter_interpolation(
		self,
		fits: Sequence[DistributionFit],
		selected_distribution: str,
		save_path: Path | str,
		dpi: int = 150,
	) -> Path:
		"""Plot the assumed linear change in selected distribution parameters."""

		parameter_names = self._distribution_parameter_names(selected_distribution)
		speeds = np.linspace(0.5, 1.0, 101)
		fig, axes = plt.subplots(2, len(parameter_names), figsize=(5.4 * len(parameter_names), 7.2), squeeze=False)

		for row, metric in enumerate(self.RELIABILITY_METRICS):
			interpolated = np.vstack([
				self.interpolate_distribution_parameters(
					fits, selected_distribution, metric, speed
				)
				for speed in speeds
			])
			for column, parameter_name in enumerate(parameter_names):
				axis = axes[row, column]
				axis.plot(speeds, interpolated[:, column], lw=1.5)
				axis.scatter(
					[0.5, 1.0],
					[interpolated[0, column], interpolated[-1, column]],
					zorder=3,
				)
				axis.set_title(f"{metric.title()}: {parameter_name}")
				axis.set_xlabel("Speed scale factor")
				axis.set_ylabel("Parameter value")
				axis.grid(alpha=0.3)

		fig.suptitle(
			f"Linear speed interpolation of {selected_distribution} parameters",
			fontsize=12,
		)
		fig.tight_layout(rect=[0, 0, 1, 0.95])
		out_path = Path(save_path)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=dpi)
		plt.close(fig)
		return out_path

	def plot_reliability_curves(
		self,
		fits: Sequence[DistributionFit],
		selected_distribution: str,
		metric: str,
		tolerances: Sequence[float],
		required_reliabilities: Sequence[float],
		save_path: Path | str,
		log_y: bool = False,
		dpi: int = 150,
	) -> Path:
		"""Plot pointwise reliability against interpolated speed scale."""

		if metric not in self.RELIABILITY_METRICS:
			raise ValueError(f"Unknown metric: {metric}")
		speeds = np.linspace(0.5, 1.0, 251)
		fig, axis = plt.subplots(figsize=(8.0, 5.0))
		curves = []
		for tolerance in tolerances:
			reliability = self.reliability_over_speed(
				fits, selected_distribution, metric, speeds, float(tolerance)
			)
			curves.append((float(tolerance), reliability))

		positive_series = [
			reliability[reliability > 0.0]
			for _, reliability in curves
			if np.any(reliability > 0.0)
		]
		positive_values = np.concatenate(positive_series) if positive_series else np.array([])
		log_floor = max(float(positive_values.min()) * 0.8, 1e-12) if len(positive_values) else 1e-12

		for tolerance, reliability in curves:
			plot_values = np.maximum(reliability, log_floor) if log_y else reliability
			axis.plot(speeds, plot_values, lw=1.6, label=f"Tolerance={tolerance:g}")

		for requirement in required_reliabilities:
			axis.axhline(
				float(requirement),
				color="0.25",
				lw=0.8,
				ls="--",
				alpha=0.65,
			)
			axis.text(1.002, requirement, f"{requirement:.0%}", va="center", fontsize=8)

		axis.set_xlim(0.5, 1.0)
		if log_y:
			axis.set_yscale("log")
			axis.set_ylim(log_floor, 1.01)
		else:
			axis.set_ylim(0.0, 1.01)
		axis.set_xlabel("Speed scale factor")
		axis.set_ylabel("Reliability: P(error <= tolerance)")
		scale_label = ", log reliability scale" if log_y else ""
		axis.set_title(
			f"{metric.title()}-error reliability curves "
			f"({selected_distribution}{scale_label})"
		)
		axis.legend()
		axis.grid(alpha=0.3)
		fig.tight_layout()
		out_path = Path(save_path)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=dpi)
		plt.close(fig)
		return out_path

	def plot_maximum_speed_by_tolerance(
		self,
		results: pd.DataFrame,
		metric: str,
		save_path: Path | str,
		dpi: int = 150,
	) -> Path:
		"""Plot one metric's maximum allowable speed against its tolerance."""

		fig, axis = plt.subplots(figsize=(7.5, 5.0))
		for required_reliability, group in results.groupby("required_reliability", sort=True):
			group = group.sort_values("tolerance")
			axis.plot(
				group["tolerance"],
				group["maximum_speed_scale"],
				marker="o",
				lw=1.5,
				label=f"Required reliability={required_reliability:.0%}",
			)

		unit = "mm" if metric == "position" else "deg"
		axis.set_ylim(0.48, 1.02)
		axis.set_xlabel(f"{metric.title()} error tolerance ({unit})")
		axis.set_ylabel("Maximum allowable speed scale")
		axis.set_title(f"{metric.title()}-based maximum allowable speed")
		axis.legend()
		axis.grid(alpha=0.3)

		fig.tight_layout()
		out_path = Path(save_path)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=dpi)
		plt.close(fig)
		return out_path

	def run_speed_reliability_analysis(
		self,
		position_tolerances_mm: Sequence[float] = (0.2, 0.3, 0.5, 0.7, 1.0),
		orientation_tolerances_deg: Sequence[float] = (0.02, 0.03, 0.05, 0.07, 0.1),
		required_reliabilities: Sequence[float] = (0.90, 0.95, 0.99),
		output_dir: Optional[Path | str] = None,
		dpi: int = 150,
	) -> Dict[str, object]:
		"""Run the complete distribution-based speed-limit analysis pipeline."""

		out_dir = Path(output_dir) if output_dir is not None else self.project_root / "output" / "reliability"
		out_dir.mkdir(parents=True, exist_ok=True)
		fits, selected_distribution = self.fit_reliability_distributions(
			probability_plot_dir=out_dir / "probability_plots",
			distribution_plot_dir=out_dir / "distribution_fits",
			dpi=dpi,
		)

		selected_rows = []
		for fit in fits:
			if fit.distribution != selected_distribution:
				continue
			row = {
				"metric": fit.metric,
				"speed_scale": fit.speed_scale,
				"distribution": fit.distribution,
				"probability_plot_r2": fit.probability_plot_r2,
				"aic": fit.aic,
			}
			for name, value in zip(
				self._distribution_parameter_names(selected_distribution),
				fit.parameters,
			):
				row[name] = value
			selected_rows.append(row)
		selected_parameter_table = pd.DataFrame(selected_rows)
		selected_parameter_table.to_csv(out_dir / "selected_model_parameters.csv", index=False)
		(out_dir / "selected_model.txt").write_text(
			f"Selected common distribution: {selected_distribution}\n"
			"Selection criterion: highest mean probability-plot R-squared across "
			"position/orientation and speed 0.5/1.0 datasets.\n",
			encoding="utf-8",
		)

		self.plot_selected_parameter_interpolation(
			fits,
			selected_distribution,
			out_dir / "selected_parameter_interpolation.png",
			dpi=dpi,
		)
		self.plot_reliability_curves(
			fits,
			selected_distribution,
			"position",
			position_tolerances_mm,
			required_reliabilities,
			out_dir / "position_reliability_curves.png",
			dpi=dpi,
		)
		self.plot_reliability_curves(
			fits,
			selected_distribution,
			"orientation",
			orientation_tolerances_deg,
			required_reliabilities,
			out_dir / "orientation_reliability_curves.png",
			dpi=dpi,
		)
		self.plot_reliability_curves(
			fits,
			selected_distribution,
			"position",
			position_tolerances_mm,
			required_reliabilities,
			out_dir / "position_reliability_curves_log_scale.png",
			log_y=True,
			dpi=dpi,
		)
		self.plot_reliability_curves(
			fits,
			selected_distribution,
			"orientation",
			orientation_tolerances_deg,
			required_reliabilities,
			out_dir / "orientation_reliability_curves_log_scale.png",
			log_y=True,
			dpi=dpi,
		)

		position_results = pd.DataFrame([
			self.maximum_allowable_speed(
				fits,
				selected_distribution,
				"position",
				float(position_tolerance),
				float(required_reliability),
			)
			for required_reliability in required_reliabilities
			for position_tolerance in position_tolerances_mm
		])
		orientation_results = pd.DataFrame([
			self.maximum_allowable_speed(
				fits,
				selected_distribution,
				"orientation",
				float(orientation_tolerance),
				float(required_reliability),
			)
			for required_reliability in required_reliabilities
			for orientation_tolerance in orientation_tolerances_deg
		])
		position_results.to_csv(out_dir / "position_maximum_speed_table.csv", index=False)
		orientation_results.to_csv(out_dir / "orientation_maximum_speed_table.csv", index=False)

		self.plot_maximum_speed_by_tolerance(
			position_results,
			"position",
			out_dir / "position_maximum_speed_by_tolerance.png",
			dpi=dpi,
		)
		self.plot_maximum_speed_by_tolerance(
			orientation_results,
			"orientation",
			out_dir / "orientation_maximum_speed_by_tolerance.png",
			dpi=dpi,
		)

		position_for_merge = position_results.rename(columns={
			"tolerance": "position_tolerance_mm",
			"maximum_speed_scale": "position_maximum_speed_scale",
			"reliability_at_limit": "position_reliability_at_limit",
			"status": "position_status",
		}).drop(columns=["metric", "tolerance_unit"])
		orientation_for_merge = orientation_results.rename(columns={
			"tolerance": "orientation_tolerance_deg",
			"maximum_speed_scale": "orientation_maximum_speed_scale",
			"reliability_at_limit": "orientation_reliability_at_limit",
			"status": "orientation_status",
		}).drop(columns=["metric", "tolerance_unit"])
		practical_results = position_for_merge.merge(
			orientation_for_merge,
			on="required_reliability",
			how="outer",
		)
		practical_results["recommended_speed_scale"] = practical_results[[
			"position_maximum_speed_scale",
			"orientation_maximum_speed_scale",
		]].min(axis=1, skipna=False)
		practical_results["limiting_metric"] = np.where(
			practical_results["position_maximum_speed_scale"]
			<= practical_results["orientation_maximum_speed_scale"],
			"position",
			"orientation",
		)
		practical_results.loc[
			practical_results["recommended_speed_scale"].isna(),
			"limiting_metric",
		] = "not_feasible"
		practical_results.to_csv(out_dir / "practical_recommended_speed_table.csv", index=False)

		return {
			"selected_distribution": selected_distribution,
			"fits": fits,
			"position_speed_results": position_results,
			"orientation_speed_results": orientation_results,
			"practical_speed_results": practical_results,
			"output_dir": out_dir,
		}

	@staticmethod
	def _bootstrap_relative_effect(
		baseline: np.ndarray,
		comparison: np.ndarray,
		rng: np.random.Generator,
		bootstrap_samples: int,
	) -> tuple[float, float, float]:
		"""Return relative mean change and a run-level bootstrap interval."""

		baseline = np.asarray(baseline, dtype=float)
		comparison = np.asarray(comparison, dtype=float)
		estimate = (comparison.mean() / baseline.mean() - 1.0) * 100.0
		bootstrap_effects = np.empty(bootstrap_samples, dtype=float)

		for index in range(bootstrap_samples):
			baseline_sample = rng.choice(baseline, size=len(baseline), replace=True)
			comparison_sample = rng.choice(comparison, size=len(comparison), replace=True)
			bootstrap_effects[index] = (
				comparison_sample.mean() / baseline_sample.mean() - 1.0
			) * 100.0

		lower, upper = np.quantile(bootstrap_effects, [0.025, 0.975])
		return float(estimate), float(lower), float(upper)

	def run_payload_sensitivity_analysis(
		self,
		position_tolerances_mm: Sequence[float] = tuple(i * 0.1 for i in range(1, 11)),
		orientation_tolerances_deg: Sequence[float] = tuple(i * 0.01 for i in range(1, 11)),
		required_reliabilities: Sequence[float] = (0.90, 0.95, 0.99),
		output_dir: Optional[Path | str] = None,
		bootstrap_samples: int = 5000,
		random_seed: int = 2026,
		dpi: int = 150,
	) -> Dict[str, object]:
		"""Compare payload effects with speed effects using run-level summaries.

		This is a sensitivity analysis, not a proof that payload has zero effect.
		Each CSV file is treated as one replicate to avoid presenting correlated
		8 ms samples as independent experimental replicates.
		"""

		out_dir = Path(output_dir) if output_dir is not None else self.project_root / "output" / "payload_sensitivity"
		out_dir.mkdir(parents=True, exist_ok=True)
		df = self._normal_start_reliability_frame()

		metric_columns = {
			"position": "FK_POSITION_ERROR_MM",
			"orientation": "FK_ORIENTATION_ERROR_DEG",
		}
		statistics = {
			"mean": lambda values: values.mean(),
			"median": lambda values: values.median(),
			"p95": lambda values: values.quantile(0.95),
			"p99": lambda values: values.quantile(0.99),
		}

		run_rows = []
		for keys, group in df.groupby(
			["file_name", "speed_scale", "payload_lb", "replicate"],
			sort=True,
		):
			file_name, speed_scale, payload_lb, replicate = keys
			for metric, column in metric_columns.items():
				values = pd.to_numeric(group[column], errors="coerce").dropna()
				row = {
					"file_name": file_name,
					"speed_scale": float(speed_scale),
					"payload_lb": float(payload_lb),
					"replicate": int(replicate),
					"metric": metric,
					"sample_count": len(values),
				}
				for statistic, function in statistics.items():
					row[statistic] = float(function(values))
				run_rows.append(row)

		run_summary = pd.DataFrame(run_rows)
		run_summary.to_csv(out_dir / "run_level_error_summary.csv", index=False)

		cell_summary = (
			run_summary.groupby(["metric", "speed_scale", "payload_lb"], as_index=False)
			.agg(
				run_count=("file_name", "count"),
				mean_of_run_means=("mean", "mean"),
				mean_of_run_medians=("median", "mean"),
				mean_of_run_p95=("p95", "mean"),
				mean_of_run_p99=("p99", "mean"),
			)
		)
		cell_summary.to_csv(out_dir / "condition_error_summary.csv", index=False)

		rng = np.random.default_rng(random_seed)
		effect_rows = []
		for metric in metric_columns:
			metric_df = run_summary[run_summary["metric"] == metric]
			for statistic in statistics:
				for speed_scale in sorted(metric_df["speed_scale"].unique()):
					baseline = metric_df[
						(metric_df["speed_scale"] == speed_scale)
						& np.isclose(metric_df["payload_lb"], 1.6)
					][statistic].to_numpy()
					comparison = metric_df[
						(metric_df["speed_scale"] == speed_scale)
						& np.isclose(metric_df["payload_lb"], 4.5)
					][statistic].to_numpy()
					estimate, lower, upper = self._bootstrap_relative_effect(
						baseline, comparison, rng, bootstrap_samples
					)
					effect_rows.append({
						"metric": metric,
						"statistic": statistic,
						"effect_type": "payload_1.6_to_4.5_lb",
						"fixed_factor": "speed_scale",
						"fixed_level": float(speed_scale),
						"relative_change_percent": estimate,
						"ci95_lower_percent": lower,
						"ci95_upper_percent": upper,
					})

				for payload_lb in sorted(metric_df["payload_lb"].unique()):
					baseline = metric_df[
						(metric_df["payload_lb"] == payload_lb)
						& np.isclose(metric_df["speed_scale"], 0.5)
					][statistic].to_numpy()
					comparison = metric_df[
						(metric_df["payload_lb"] == payload_lb)
						& np.isclose(metric_df["speed_scale"], 1.0)
					][statistic].to_numpy()
					estimate, lower, upper = self._bootstrap_relative_effect(
						baseline, comparison, rng, bootstrap_samples
					)
					effect_rows.append({
						"metric": metric,
						"statistic": statistic,
						"effect_type": "speed_0.5_to_1.0",
						"fixed_factor": "payload_lb",
						"fixed_level": float(payload_lb),
						"relative_change_percent": estimate,
						"ci95_lower_percent": lower,
						"ci95_upper_percent": upper,
					})

		effect_summary = pd.DataFrame(effect_rows)
		effect_summary.to_csv(out_dir / "payload_vs_speed_effects.csv", index=False)

		fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
		for row, metric in enumerate(metric_columns):
			metric_df = run_summary[run_summary["metric"] == metric]
			for column, statistic in enumerate(["median", "p95"]):
				axis = axes[row, column]
				for speed_scale, marker, color in [
					(0.5, "o", "tab:blue"),
					(1.0, "s", "tab:orange"),
				]:
					subset = metric_df[metric_df["speed_scale"] == speed_scale]
					for payload_lb in [1.6, 4.5]:
						points = subset[np.isclose(subset["payload_lb"], payload_lb)][statistic]
						axis.scatter(
							np.full(len(points), payload_lb),
							points,
							marker=marker,
							color=color,
							alpha=0.55,
						)
					means = subset.groupby("payload_lb")[statistic].mean().sort_index()
					axis.plot(
						means.index,
						means.values,
						marker=marker,
						color=color,
						lw=1.6,
						label=f"speed={speed_scale:g}",
					)
				axis.set_xticks([1.6, 4.5])
				axis.set_xlabel("Payload (lb)")
				axis.set_ylabel(self.RELIABILITY_METRICS[metric][1])
				axis.set_title(f"{metric.title()} run-level {statistic}")
				axis.grid(alpha=0.3)
				axis.legend()
		fig.suptitle("Payload sensitivity at fixed speed (normal-start runs)", fontsize=12)
		fig.tight_layout(rect=[0, 0, 1, 0.96])
		fig.savefig(out_dir / "payload_run_level_comparison.png", dpi=dpi)
		plt.close(fig)

		selected_effects = effect_summary[effect_summary["statistic"].isin(["median", "p95", "p99"])]
		fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), sharex=False)
		for row, metric in enumerate(metric_columns):
			for column, statistic in enumerate(["median", "p95", "p99"]):
				axis = axes[row, column]
				subset = selected_effects[
					(selected_effects["metric"] == metric)
					& (selected_effects["statistic"] == statistic)
				].copy()
				subset["label"] = subset.apply(
					lambda item: (
						f"Payload effect | speed={item['fixed_level']:g}"
						if item["effect_type"].startswith("payload")
						else f"Speed effect | payload={item['fixed_level']:g} lb"
					),
					axis=1,
				)
				y = np.arange(len(subset))
				x = subset["relative_change_percent"].to_numpy()
				xerr = np.vstack([
					np.maximum(0.0, x - subset["ci95_lower_percent"].to_numpy()),
					np.maximum(0.0, subset["ci95_upper_percent"].to_numpy() - x),
				])
				colors = [
					"tab:orange" if effect.startswith("payload") else "tab:blue"
					for effect in subset["effect_type"]
				]
				for point_index, color in enumerate(colors):
					axis.errorbar(
						x[point_index],
						y[point_index],
						xerr=xerr[:, point_index].reshape(2, 1),
						fmt="o",
						color=color,
						ecolor=color,
						capsize=3,
						alpha=0.85,
					)
				axis.axvline(0.0, color="black", lw=0.8, ls="--")
				axis.set_yticks(y, subset["label"], fontsize=8)
				axis.set_xlabel("Relative change (%)")
				axis.set_title(f"{metric.title()} {statistic}")
				axis.grid(axis="x", alpha=0.3)
		fig.suptitle("Payload effects versus speed effects (run bootstrap 95% intervals)", fontsize=12)
		fig.tight_layout(rect=[0, 0, 1, 0.96])
		fig.savefig(out_dir / "payload_vs_speed_effect_comparison.png", dpi=dpi)
		plt.close(fig)

		fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
		for row, (metric, column_name) in enumerate(metric_columns.items()):
			for column, speed_scale in enumerate([0.5, 1.0]):
				axis = axes[row, column]
				for payload_lb in [1.6, 4.5]:
					values = np.sort(pd.to_numeric(
						df[
							(np.isclose(df["speed_scale"], speed_scale))
							& (np.isclose(df["payload_lb"], payload_lb))
						][column_name],
						errors="coerce",
					).dropna().to_numpy())
					probability = np.arange(1, len(values) + 1) / len(values)
					axis.plot(values, probability, lw=1.3, label=f"{payload_lb:g} lb")
				axis.set_xlim(0.0, df[df["speed_scale"] == speed_scale][column_name].quantile(0.995))
				axis.set_xlabel(self.RELIABILITY_METRICS[metric][1])
				axis.set_ylabel("Empirical CDF")
				axis.set_title(f"{metric.title()} | speed={speed_scale:g}")
				axis.legend()
				axis.grid(alpha=0.3)
		fig.suptitle("Payload-specific empirical error distributions", fontsize=12)
		fig.tight_layout(rect=[0, 0, 1, 0.96])
		fig.savefig(out_dir / "payload_empirical_cdf.png", dpi=dpi)
		plt.close(fig)

		selected_distribution = "lognormal"
		speed_grid = np.linspace(0.5, 1.0, 5001)

		def fitted_parameters(metric: str, speed_scale: float, payload_lb: Optional[float]) -> np.ndarray:
			column = metric_columns[metric]
			subset = df[np.isclose(df["speed_scale"], speed_scale)]
			if payload_lb is not None:
				subset = subset[np.isclose(subset["payload_lb"], payload_lb)]
			values = pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy()
			return np.asarray(
				self._fit_distribution_values(values, selected_distribution),
				dtype=float,
			)

		def payload_model_speed_limit(
			metric: str,
			tolerance: float,
			required_reliability: float,
			payload_lb: Optional[float],
		) -> tuple[float, float, str]:
			low = fitted_parameters(metric, 0.5, payload_lb)
			high = fitted_parameters(metric, 1.0, payload_lb)
			weights = ((speed_grid - 0.5) / 0.5)[:, None]
			parameters = low + weights * (high - low)
			reliability = stats.lognorm.cdf(
				tolerance,
				s=parameters[:, 0],
				loc=0.0,
				scale=parameters[:, 1],
			)
			feasible = np.flatnonzero(reliability >= required_reliability)
			if not len(feasible):
				return float("nan"), float(reliability[0]), "not_feasible_at_0.5"
			index = int(feasible[-1])
			return float(speed_grid[index]), float(reliability[index]), "feasible"

		model_rows = []
		for metric, tolerances in [
			("position", position_tolerances_mm),
			("orientation", orientation_tolerances_deg),
		]:
			for required_reliability in required_reliabilities:
				for tolerance in tolerances:
					for payload_label, payload_lb in [
						("pooled", None),
						("1.6 lb", 1.6),
						("4.5 lb", 4.5),
					]:
						maximum_speed, reliability_at_limit, status = payload_model_speed_limit(
							metric,
							float(tolerance),
							float(required_reliability),
							payload_lb,
						)
						model_rows.append({
							"metric": metric,
							"tolerance": float(tolerance),
							"required_reliability": float(required_reliability),
							"payload_model": payload_label,
							"maximum_speed_scale": maximum_speed,
							"reliability_at_limit": reliability_at_limit,
							"status": status,
						})

		payload_model_limits = pd.DataFrame(model_rows)
		payload_model_limits.to_csv(out_dir / "payload_specific_speed_limits.csv", index=False)

		fig, axes = plt.subplots(2, len(required_reliabilities), figsize=(15.0, 7.5), sharey=True)
		for row, metric in enumerate(["position", "orientation"]):
			for column, required_reliability in enumerate(required_reliabilities):
				axis = axes[row, column]
				subset = payload_model_limits[
					(payload_model_limits["metric"] == metric)
					& np.isclose(payload_model_limits["required_reliability"], required_reliability)
				]
				for payload_model, style in [
					("pooled", {"color": "black", "ls": "--"}),
					("1.6 lb", {"color": "tab:blue", "ls": "-"}),
					("4.5 lb", {"color": "tab:orange", "ls": "-"}),
				]:
					group = subset[subset["payload_model"] == payload_model].sort_values("tolerance")
					axis.plot(
						group["tolerance"],
						group["maximum_speed_scale"],
						marker="o",
						label=payload_model,
						**style,
					)
				axis.set_ylim(0.48, 1.02)
				axis.set_xlabel(
					"Position tolerance (mm)"
					if metric == "position"
					else "Orientation tolerance (deg)"
				)
				axis.set_ylabel("Maximum speed scale")
				axis.set_title(f"{metric.title()} | reliability={required_reliability:.0%}")
				axis.grid(alpha=0.3)
				axis.legend(fontsize=8)
		fig.suptitle("Sensitivity of recommended speed to payload pooling", fontsize=12)
		fig.tight_layout(rect=[0, 0, 1, 0.96])
		fig.savefig(out_dir / "payload_specific_speed_limit_comparison.png", dpi=dpi)
		plt.close(fig)

		return {
			"run_summary": run_summary,
			"condition_summary": cell_summary,
			"effect_summary": effect_summary,
			"payload_specific_speed_limits": payload_model_limits,
			"output_dir": out_dir,
		}

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

	# 6. Distribution-based reliability and maximum allowable speed analysis
	# reliability_results = analyzer.run_speed_reliability_analysis()
	reliability_results = analyzer.run_speed_reliability_analysis(
		position_tolerances_mm=[i * 0.1 for i in range(1, 11)],
		orientation_tolerances_deg=[i * 0.01 for i in range(1, 11)],
		required_reliabilities=[0.90, 0.95, 0.99],
	)
	payload_results = analyzer.run_payload_sensitivity_analysis()

	print(f"Saved single-file signed plots: {len(single_paths)}")
	print(f"Saved single-file absolute plots: {len(single_abs_paths)}")
	print(f"Saved factor-effect plots: {len(factor_paths)}")
	print(f"Saved Cartesian single-file plots: {len(cartesian_paths)}")
	print(f"Saved Cartesian factor-effect plots: {len(cartesian_factor_paths)}")
	print(f"Selected reliability model: {reliability_results['selected_distribution']}")
	print(f"Saved reliability outputs: {reliability_results['output_dir']}")
	print(f"Saved payload sensitivity outputs: {payload_results['output_dir']}")


if __name__ == "__main__":
	main()
