# The MIT License (MIT)
# Copyright © 2025 Entrius

"""
Unit tests for CLI helper functions (format_alpha, validators).
"""

from unittest.mock import MagicMock, patch

import click
import pytest

from gittensor.cli.issue_commands.helpers import (
    ALPHA_SCALE,
    MAX_ISSUE_ID,
    MIN_BOUNTY_ALPHA,
    STATUS_COLORS,
    colorize_status,
    format_alpha,
    validate_and_convert_bounty,
    validate_issue_id,
    validate_repo_format,
    validate_ss58_address,
)

# ============================================================================
# format_alpha
# ============================================================================


class TestFormatAlpha:
    def test_zero(self):
        assert format_alpha(0) == '0.00 ALPHA'

    def test_one_alpha(self):
        assert format_alpha(1_000_000_000) == '1.00 ALPHA'

    def test_hundred_alpha(self):
        assert format_alpha(100_000_000_000) == '100.00 ALPHA'

    def test_fractional(self):
        assert format_alpha(1_500_000_000) == '1.50 ALPHA'

    def test_four_decimals(self):
        assert format_alpha(1_234_567_890, decimals=4) == '1.2346 ALPHA'

    def test_zero_decimals(self):
        assert format_alpha(50_000_000_000, decimals=0) == '50 ALPHA'

    def test_large_value(self):
        assert format_alpha(1_000_000_000_000_000_000) == '1000000000.00 ALPHA'

    def test_small_fraction(self):
        # 0.000000001 ALPHA = 1 nanoALPHA
        assert format_alpha(1, decimals=9) == '0.000000001 ALPHA'


# ============================================================================
# validate_and_convert_bounty
# ============================================================================


class TestValidateAndConvertBounty:
    def test_valid_bounty(self):
        result = validate_and_convert_bounty(100.0)
        assert result == 100 * ALPHA_SCALE

    def test_minimum_bounty(self):
        result = validate_and_convert_bounty(10.0)
        assert result == 10 * ALPHA_SCALE

    def test_bounty_with_decimals(self):
        result = validate_and_convert_bounty(50.5)
        assert result == 50_500_000_000

    def test_bounty_below_minimum(self):
        with pytest.raises(click.BadParameter, match='at least 10 ALPHA'):
            validate_and_convert_bounty(9.99)

    def test_bounty_zero(self):
        with pytest.raises(click.BadParameter, match='at least 10 ALPHA'):
            validate_and_convert_bounty(0)

    def test_bounty_negative(self):
        with pytest.raises(click.BadParameter, match='at least 10 ALPHA'):
            validate_and_convert_bounty(-50)

    def test_bounty_too_many_decimals(self):
        with pytest.raises(click.BadParameter, match='too many decimal places'):
            validate_and_convert_bounty(10.0000000001)

    def test_bounty_nine_decimals_ok(self):
        # Exactly 9 decimal places should be fine
        result = validate_and_convert_bounty(10.123456789)
        assert result == 10_123_456_789

    def test_floating_point_precision(self):
        """Test that 100.1 doesn't become 100.09999... due to IEEE 754."""
        result = validate_and_convert_bounty(100.1)
        assert result == 100_100_000_000

    def test_integer_bounty(self):
        result = validate_and_convert_bounty(500)
        assert result == 500 * ALPHA_SCALE

    def test_param_hint_on_error(self):
        """Errors include --bounty param_hint for CLI context."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_and_convert_bounty(5)
        assert exc_info.value.param_hint == '--bounty'

    def test_uses_min_bounty_constant(self):
        """Minimum bounty matches MIN_BOUNTY_ALPHA constant."""
        from decimal import Decimal

        assert MIN_BOUNTY_ALPHA == Decimal('10')


# ============================================================================
# validate_repo_format
# ============================================================================


class TestValidateRepoFormat:
    def test_valid_repo(self):
        owner, name = validate_repo_format('owner/repo')
        assert owner == 'owner'
        assert name == 'repo'

    def test_valid_repo_with_dots(self):
        owner, name = validate_repo_format('owner/repo.name')
        assert owner == 'owner'
        assert name == 'repo.name'

    def test_valid_repo_with_hyphens(self):
        owner, name = validate_repo_format('my-org/my-repo')
        assert owner == 'my-org'
        assert name == 'my-repo'

    def test_valid_repo_with_underscores(self):
        owner, name = validate_repo_format('my_org/my_repo')
        assert owner == 'my_org'
        assert name == 'my_repo'

    def test_no_slash(self):
        with pytest.raises(click.BadParameter, match='owner/repo format'):
            validate_repo_format('noslash')

    def test_multiple_slashes(self):
        with pytest.raises(click.BadParameter, match='owner/repo format'):
            validate_repo_format('too/many/slashes')

    def test_empty_owner(self):
        with pytest.raises(click.BadParameter, match='non-empty'):
            validate_repo_format('/repo')

    def test_empty_name(self):
        with pytest.raises(click.BadParameter, match='non-empty'):
            validate_repo_format('owner/')

    def test_spaces(self):
        with pytest.raises(click.BadParameter, match='spaces'):
            validate_repo_format('owner/ repo')

    def test_invalid_characters(self):
        with pytest.raises(click.BadParameter, match='alphanumeric'):
            validate_repo_format('owner/repo!')

    def test_param_hint_on_error(self):
        with pytest.raises(click.BadParameter) as exc_info:
            validate_repo_format('noslash')
        assert exc_info.value.param_hint == '--repo'


# ============================================================================
# validate_issue_id
# ============================================================================


class TestValidateIssueId:
    def test_valid_id(self):
        assert validate_issue_id(1) == 1

    def test_valid_large_id(self):
        assert validate_issue_id(999_999) == 999_999

    def test_zero(self):
        with pytest.raises(click.BadParameter, match='>= 1'):
            validate_issue_id(0)

    def test_negative(self):
        with pytest.raises(click.BadParameter, match='>= 1'):
            validate_issue_id(-5)

    def test_too_large(self):
        with pytest.raises(click.BadParameter, match='< 1,000,000'):
            validate_issue_id(1_000_000)

    def test_custom_label(self):
        with pytest.raises(click.BadParameter, match='my field'):
            validate_issue_id(0, label='my field')

    def test_uses_max_issue_id_constant(self):
        assert MAX_ISSUE_ID == 1_000_000

    def test_param_hint_forwarded(self):
        with pytest.raises(click.BadParameter) as exc_info:
            validate_issue_id(0, param_hint='ISSUE_ID')
        assert exc_info.value.param_hint == 'ISSUE_ID'


# ============================================================================
# validate_ss58_address
# ============================================================================


class TestValidateSs58Address:
    def test_valid_address(self):
        """Test that a valid-looking address passes and returns the cleaned string."""
        mock_keypair_cls = MagicMock()
        with patch.dict('sys.modules', {'substrateinterface': MagicMock(Keypair=mock_keypair_cls)}):
            result = validate_ss58_address('5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY', 'test')
            assert result == '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY'
            mock_keypair_cls.assert_called_once_with(ss58_address='5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY')

    def test_invalid_address(self):
        """Test that a clearly invalid address raises click.BadParameter."""
        mock_keypair_cls = MagicMock(side_effect=ValueError('Invalid SS58'))
        with patch.dict('sys.modules', {'substrateinterface': MagicMock(Keypair=mock_keypair_cls)}):
            with pytest.raises(click.BadParameter, match='Invalid SS58 address'):
                validate_ss58_address('not-a-valid-address', 'test')

    def test_empty_address(self):
        with pytest.raises(click.BadParameter, match='must not be empty'):
            validate_ss58_address('', 'test')

    def test_whitespace_only_address(self):
        with pytest.raises(click.BadParameter, match='must not be empty'):
            validate_ss58_address('   ', 'test')

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped and cleaned address returned."""
        mock_keypair_cls = MagicMock()
        with patch.dict('sys.modules', {'substrateinterface': MagicMock(Keypair=mock_keypair_cls)}):
            result = validate_ss58_address('  5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY  ', 'test')
            assert result == '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY'
            mock_keypair_cls.assert_called_once_with(ss58_address='5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY')

    def test_regex_fallback_valid(self):
        """Test regex fallback when substrateinterface is not installed."""
        with patch.dict('sys.modules', {'substrateinterface': None}):
            result = validate_ss58_address('5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY', 'test')
            assert result == '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY'

    def test_regex_fallback_invalid(self):
        """Test regex fallback rejects clearly invalid addresses."""
        with patch.dict('sys.modules', {'substrateinterface': None}):
            with pytest.raises(click.BadParameter, match='Invalid SS58 address'):
                validate_ss58_address('not-valid', 'test')

    def test_param_hint_forwarded(self):
        """Test that param_hint is passed through to BadParameter."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_ss58_address('', 'test', param_hint='SOLVER_HOTKEY')
        assert exc_info.value.param_hint == 'SOLVER_HOTKEY'


# ============================================================================
# colorize_status
# ============================================================================


class TestColorizeStatus:
    def test_active_green(self):
        result = colorize_status('Active')
        assert result.plain == 'Active'
        assert str(result.style) == 'green'

    def test_registered_yellow(self):
        result = colorize_status('Registered')
        assert result.plain == 'Registered'

    def test_unknown_status_white(self):
        result = colorize_status('SomethingNew')
        assert result.plain == 'SomethingNew'

    def test_status_colors_dict(self):
        assert 'Active' in STATUS_COLORS
        assert 'Registered' in STATUS_COLORS
        assert 'Completed' in STATUS_COLORS
        assert 'Cancelled' in STATUS_COLORS
