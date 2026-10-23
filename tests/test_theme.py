from pathlib import Path
import unittest

import streamlit as st
import toml


CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def luminance(color):
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


def contrast(foreground, background):
    low, high = sorted((luminance(foreground), luminance(background)))
    return (high + 0.05) / (low + 0.05)


def composite(foreground, background, opacity):
    return "#" + "".join(
        f"{round(int(foreground[i:i + 2], 16) * opacity + int(background[i:i + 2], 16) * (1 - opacity)):02X}"
        for i in (1, 3, 5)
    )


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = toml.load(CONFIG)

    def assert_contrast(self, foreground, background, minimum):
        self.assertGreaterEqual(
            contrast(foreground, background), minimum,
            f"{foreground} on {background} must have at least {minimum}:1 contrast",
        )

    def test_every_config_option_is_supported_and_loaded(self):
        def check_options(section, prefix=""):
            for key, value in section.items():
                option = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    check_options(value, option)
                else:
                    with self.subTest(option=option):
                        self.assertEqual(st.get_option(option), value)

        check_options(self.config)

    def test_text_links_controls_and_tables_have_accessible_contrast(self):
        for mode in ("light", "dark"):
            main = self.config["theme"][mode]
            for surface, palette in (("main", main), ("sidebar", {**main, **main["sidebar"]})):
                with self.subTest(mode=mode, surface=surface):
                    for background in ("backgroundColor", "secondaryBackgroundColor"):
                        for foreground in ("textColor", "linkColor", "codeTextColor"):
                            self.assert_contrast(palette[foreground], palette[background], 4.5)
                        self.assert_contrast(palette["primaryColor"], palette[background], 3)
                        self.assert_contrast(palette["borderColor"], palette[background], 3)
                        self.assert_contrast(
                            composite(palette["textColor"], palette[background], 0.6),
                            palette[background], 4.5,
                        )
                    self.assert_contrast(palette["primaryColor"], palette["backgroundColor"], 4.5)
                    self.assert_contrast("#FFFFFF", palette["primaryColor"], 4.5)
                    self.assert_contrast(
                        composite(palette["textColor"], palette["dataframeHeaderBackgroundColor"], 0.6),
                        palette["dataframeHeaderBackgroundColor"], 4.5,
                    )

    def test_status_messages_have_accessible_contrast(self):
        for mode in ("light", "dark"):
            palette = self.config["theme"][mode]
            for status in ("blue", "green", "yellow", "red"):
                with self.subTest(mode=mode, status=status):
                    self.assert_contrast(
                        palette[f"{status}TextColor"], palette[f"{status}BackgroundColor"], 4.5
                    )

    def test_native_chart_label_colors_meet_large_text_contrast(self):
        # Streamlit 1.51 supplies these colors, not the configurable textColor.
        for mode, color in (("light", "#808495"), ("dark", "#E6EAF1")):
            self.assert_contrast(color, self.config["theme"][mode]["backgroundColor"], 3)

    def test_chart_palettes_are_readable_on_both_canvases(self):
        theme = self.config["theme"]
        self.assertEqual(len(theme["chartCategoricalColors"]), 10)
        self.assertEqual(len(set(theme["chartCategoricalColors"])), 10)
        self.assertEqual(len(theme["chartSequentialColors"]), 10)
        for mode in ("light", "dark"):
            for color in theme["chartCategoricalColors"] + theme["chartSequentialColors"]:
                with self.subTest(mode=mode, color=color):
                    self.assert_contrast(color, theme[mode]["backgroundColor"], 3)
        levels = [luminance(color) for color in theme["chartSequentialColors"]]
        self.assertEqual(levels, sorted(set(levels)))


if __name__ == "__main__":
    unittest.main()
