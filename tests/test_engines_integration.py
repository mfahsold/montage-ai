import unittest
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath("src"))

from montage_ai.core.montage_builder import MontageBuilder
from montage_ai.core.context import MontageContext
from montage_ai.core.analysis_engine import AssetAnalyzer
from montage_ai.core.render_engine import RenderEngine
from montage_ai.core.pacing_engine import PacingEngine
from montage_ai.core.selection_engine import SelectionEngine

class TestEnginesIntegration(unittest.TestCase):
    def test_engines_initialization(self):
        """Verify that MontageBuilder initializes its sub-engines."""
        try:
            # Simple instantiation
            builder = MontageBuilder(variant_id=1)
            
            # Check context
            self.assertIsInstance(builder.ctx, MontageContext)
            
            # Check AssetAnalyzer
            self.assertTrue(hasattr(builder, '_analyzer'))
            self.assertIsInstance(builder._analyzer, AssetAnalyzer)
            self.assertEqual(builder._analyzer.ctx, builder.ctx)
            
            # Check RenderEngine
            self.assertTrue(hasattr(builder, '_render_engine'))
            self.assertIsInstance(builder._render_engine, RenderEngine)
            self.assertEqual(builder._render_engine.ctx, builder.ctx)

            # Check PacingEngine
            self.assertTrue(hasattr(builder, '_pacing_engine'))
            self.assertIsInstance(builder._pacing_engine, PacingEngine)
            self.assertEqual(builder._pacing_engine.ctx, builder.ctx)

            # Check SelectionEngine
            self.assertTrue(hasattr(builder, '_selection_engine'))
            self.assertIsInstance(builder._selection_engine, SelectionEngine)
            self.assertEqual(builder._selection_engine.ctx, builder.ctx)
            
            print("All engines initialized successfully")
            
            print("All engines initialized successfully")

        except Exception as e:
            self.fail(f"Initialization failed: {e}")

if __name__ == '__main__':
    unittest.main()


