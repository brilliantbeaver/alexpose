#!/usr/bin/env python3
"""Exercise drift, ownership, format translation and boundary protections offline."""
import copy
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import sync_agents as sync


class AgentSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        (self.root / '.agents/teams/sjepa').mkdir(parents=True)
        (self.root / '.agents/roles').mkdir()
        (self.root / '.agents/protocol.md').write_text('Common agreement.\n')
        self.roles = [
            {'id': 'research-coordinator', 'source': '.agents/roles/coordinator.md',
             'description': 'Coordinate "quoted" research.', 'coordinator': True},
            {'id': 'research-reviewer', 'source': '.agents/roles/reviewer.md',
             'description': 'Review research.', 'coordinator': False},
        ]
        for role in self.roles:
            (self.root / role['source']).write_text('Role instructions with Unicode α and a newline.\n')
        self.save_roles()
        (self.root / '.agents/teams/sjepa/roles.json').write_text('[]\n')

    def save_roles(self):
        (self.root / '.agents/general-roles.json').write_text(json.dumps(self.roles))

    def test_round_trip_and_native_worker_modes(self):
        outputs, roles = sync.render(self.root)
        self.assertEqual(len(outputs), 6)
        codex = tomllib.loads(outputs['.codex/agents/research-reviewer.toml'])
        self.assertEqual(codex['developer_instructions'], roles[1]['prompt'])
        self.assertIn('Common agreement.', codex['developer_instructions'])
        claude = outputs['.claude/agents/research-reviewer.md']
        self.assertIn('disallowedTools: Agent', claude)
        self.assertNotIn('disallowedTools', outputs['.claude/agents/research-coordinator.md'])
        self.assertIn('mode: primary', outputs['.opencode/agents/research-coordinator.md'])
        worker = outputs['.opencode/agents/research-reviewer.md']
        self.assertIn('mode: subagent', worker)
        self.assertIn('action: subagent', worker)
        self.assertIn('effect: deny', worker)
        sync.synchronize(self.root, True)
        self.assertEqual(sync.synchronize(self.root), [])

    def test_canonical_changes_report_then_update_owned_outputs(self):
        sync.synchronize(self.root, True)
        (self.root / self.roles[1]['source']).write_text('Changed canonical body.\n')
        self.assertEqual(len(sync.synchronize(self.root)), 4)
        sync.synchronize(self.root, True)
        self.assertEqual(sync.synchronize(self.root), [])

    def test_unmanaged_collision_is_detected_before_any_write(self):
        path = self.root / '.claude/agents/research-reviewer.md'
        path.parent.mkdir(parents=True)
        path.write_text('User-authored instructions.')
        with self.assertRaisesRegex(ValueError, 'unmanaged'):
            sync.synchronize(self.root, True)
        self.assertFalse((self.root / '.codex').exists())
        self.assertEqual(path.read_text(), 'User-authored instructions.')

    def test_local_edit_is_preserved(self):
        sync.synchronize(self.root, True)
        path = self.root / '.codex/agents/research-reviewer.toml'
        path.write_text(path.read_text() + '# local edit\n')
        with self.assertRaisesRegex(ValueError, 'locally edited'):
            sync.synchronize(self.root, True)
        self.assertTrue(path.read_text().endswith('# local edit\n'))

    def test_symlink_ancestor_is_rejected(self):
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (self.root / '.claude').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            sync.synchronize(self.root, True)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.root / '.codex').exists())

    def test_symlink_inventory_is_rejected(self):
        outside = Path(self.temp.name) / 'outside.json'
        outside.write_text('{"private": true}')
        (self.root / sync.STATE).symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            sync.synchronize(self.root, True)
        self.assertEqual(outside.read_text(), '{"private": true}')

    def test_duplicate_and_escaping_sources_are_rejected(self):
        self.roles.append(copy.deepcopy(self.roles[0]))
        self.save_roles()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            sync.render(self.root)
        self.roles.pop()
        self.roles[0]['source'] = '../outside.md'
        self.save_roles()
        with self.assertRaisesRegex(ValueError, 'canonical source'):
            sync.render(self.root)

    def test_legacy_sjepa_semantics_and_alias_are_preserved(self):
        legacy = {'name': 'sjepa_test', 'description': 'Preserved study role.',
                  'developer_instructions': '\nOriginal study instructions.\n'}
        source = '.agents/teams/sjepa/original.md'
        (self.root / source).write_text(legacy['developer_instructions'])
        role = {'id': 'sjepa-test', 'source': source, 'description': legacy['description'],
                'coordinator': False, 'codex_name': 'sjepa_test'}
        (self.root / '.agents/teams/sjepa/roles.json').write_text(json.dumps([role]))
        target = self.root / '.codex/agents/sjepa_test.toml'
        target.parent.mkdir(parents=True)
        target.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k, v in legacy.items()))
        sync.synchronize(self.root, True)
        self.assertEqual(tomllib.loads(target.read_text()), legacy)
        self.assertTrue((self.root / '.claude/agents/sjepa-test.md').is_file())
        self.assertIn('Host compatibility:', (self.root / '.opencode/agents/sjepa-test.md').read_text())

    def test_retired_profile_requires_explicit_review(self):
        sync.synchronize(self.root, True)
        self.roles.pop()
        self.save_roles()
        with self.assertRaisesRegex(ValueError, 'Retired'):
            sync.synchronize(self.root, True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
