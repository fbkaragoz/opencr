import pytest
from ocr_pipeline.services.output_validator import OutputValidator, ValidationStatus


@pytest.fixture
def validator():
    return OutputValidator()


class TestEmptyOutput:
    def test_empty_string(self, validator):
        result = validator.validate("", page_num=1)
        assert result.status == ValidationStatus.EMPTY

    def test_short_text(self, validator):
        result = validator.validate("hello", page_num=1)
        assert result.status == ValidationStatus.EMPTY

    def test_whitespace_only(self, validator):
        result = validator.validate("   \n\n   ", page_num=1)
        assert result.status == ValidationStatus.EMPTY


class TestPassingOutput:
    def test_normal_text(self, validator):
        text = "This is a normal paragraph of text extracted from a PDF document.\n"
        text += "It contains multiple lines of varying content.\n"
        text += "Each line is unique and meaningful."
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.PASS

    def test_markdown_content(self, validator):
        text = "# Chapter 1\n\nThis is the first paragraph.\n\n## Section 1.1\n\nMore content here."
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.PASS


class TestRepetitionDetection:
    def test_extreme_repetition_fails(self, validator):
        # 70% duplicate lines
        lines = ["Same line repeated."] * 14 + [f"Unique line {i}" for i in range(6)]
        text = "\n".join(lines)
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.FAIL
        assert any("extreme repetition" in issue for issue in result.issues)

    def test_moderate_repetition_warns(self, validator):
        # ~45% duplicate lines (above 0.35 threshold)
        lines = ["Repeated line."] * 10 + [f"Unique line {i}" for i in range(12)]
        text = "\n".join(lines)
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.WARN
        assert any("moderate repetition" in issue for issue in result.issues)

    def test_consecutive_dupes_warns(self, validator):
        lines = [f"Line {i}" for i in range(5)]
        lines += ["Identical line"] * 6  # 6 consecutive duplicates
        lines += [f"Line {i}" for i in range(10, 15)]
        text = "\n".join(lines)
        result = validator.validate(text, page_num=1)
        assert any("consecutive identical lines" in issue for issue in result.issues)


class TestCharacterDiversity:
    def test_single_char_dominance_fails(self, validator):
        # More than 50% of text is one character
        text = "a" * 100 + "bcdefgh" * 5
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.FAIL


class TestModelArtifacts:
    def test_repeated_det_tokens(self, validator):
        text = "Some text <|det|><|det|><|det|><|det|> more text after artifacts."
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.FAIL
        assert any("model artifact" in issue for issue in result.issues)

    def test_repeated_grounding_tokens(self, validator):
        text = "Text before <|grounding|><|grounding|><|grounding|> text after."
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.FAIL

    def test_repeated_word_fails(self, validator):
        text = "Normal text before " + "model " * 10 + "normal text after."
        result = validator.validate(text, page_num=1)
        assert result.status == ValidationStatus.FAIL
        assert any("model artifact" in issue for issue in result.issues)

    def test_repeated_punctuation_is_not_model_artifact(self, validator):
        text = (
            "Bu alıştırma metninde doldurma boşlukları vardır. "
            "Cümlede eksik kalan yer . . . . . . . . . . ile gösterilir. "
            "Başka bir yerde de ----------- işareti bulunur."
        )
        result = validator.validate(text, page_num=1)
        assert not any("model artifact" in issue for issue in result.issues)


class TestCorpusQualityFlags:
    def test_line_break_hyphenation_warns_with_machine_flag(self, validator):
        text = (
            "Bu metin araştırma için yeterince uzun bir OCR çıktısıdır.\n"
            "Muahedenin müba-\n"
            "delesi tarihinden itibaren sekiz ay zarfında hitama erecektir."
        )
        result = validator.validate(text, page_num=1)

        assert result.status == ValidationStatus.WARN
        assert "line_hyphenation" in result.metrics["quality_flags"]
        assert any("line-break hyphenation" in issue for issue in result.issues)

    def test_markup_leak_warns_with_machine_flag(self, validator):
        text = (
            "<center>ANKARA</center>\n"
            "Bu metin OCR motorunun temiz metne taşıdığı biçimlendirme "
            "etiketini araştırmacıya görünür kılacak kadar uzundur."
        )
        result = validator.validate(text, page_num=1)

        assert result.status == ValidationStatus.WARN
        assert "markup_leak" in result.metrics["quality_flags"]


class TestMetrics:
    def test_metrics_populated(self, validator):
        text = "Line 1\nLine 2\nLine 3\nLine 1\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9"
        result = validator.validate(text, page_num=1)
        assert "repetition_ratio" in result.metrics
        assert "total_lines" in result.metrics
        assert "unique_lines" in result.metrics
        assert "text_length" in result.metrics
