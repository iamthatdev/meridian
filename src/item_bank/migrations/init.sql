-- Item Bank Database Schema
-- PostgreSQL schema for SAT prep item management

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Domains table: defines content domains for items
CREATE TABLE IF NOT EXISTS domains (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    section VARCHAR(20) NOT NULL CHECK (section IN ('reading_writing', 'math')),
    category VARCHAR(100) NOT NULL,
    description TEXT,
    target_percentage DECIMAL(5, 2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Items table: core item data with IRT parameters
CREATE TABLE IF NOT EXISTS items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Lifecycle fields
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'pretesting', 'operational', 'retired')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Content classification
    section VARCHAR(20) NOT NULL CHECK (section IN ('reading_writing', 'math')),
    domain VARCHAR(100) NOT NULL REFERENCES domains(name) ON DELETE RESTRICT,
    difficulty VARCHAR(10) NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),

    -- IRT calibration parameters
    irt_a DECIMAL(10, 4),           -- Discrimination parameter
    irt_b DECIMAL(10, 4),           -- Difficulty parameter
    irt_c DECIMAL(10, 4),           -- Guessing parameter
    irt_source VARCHAR(50),         -- Source of calibration values
    calibrated_at TIMESTAMP WITH TIME ZONE,  -- When calibration was performed

    -- Item content (JSONB for flexibility)
    content_json JSONB NOT NULL,

    -- Auto-QA fields
    auto_qa_passed BOOLEAN NOT NULL DEFAULT FALSE,
    qa_score DECIMAL(5, 4),
    qa_flags JSONB DEFAULT '[]'::jsonb,

    -- Review tracking
    model_version VARCHAR(100),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewer_id VARCHAR(100),

    -- Retirement tracking
    retired_at TIMESTAMP WITH TIME ZONE,
    retirement_reason TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_section ON items(section);
CREATE INDEX IF NOT EXISTS idx_items_domain ON items(domain);
CREATE INDEX IF NOT EXISTS idx_items_difficulty ON items(difficulty);
CREATE INDEX IF NOT EXISTS idx_items_lifecycle ON items(status, section, domain);
CREATE INDEX IF NOT EXISTS idx_items_calibrated ON items(calibrated_at) WHERE calibrated_at IS NOT NULL;

-- Calibration log: track IRT calibration runs
CREATE TABLE IF NOT EXISTS calibration_log (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    item_count INTEGER NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    parameters JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_calibration_log_run_id ON calibration_log(run_id);
CREATE INDEX IF NOT EXISTS idx_calibration_log_status ON calibration_log(status);

-- Review records: track human review workflow
CREATE TABLE IF NOT EXISTS review_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    item_id UUID NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    reviewer_id VARCHAR(100) NOT NULL,
    review_type VARCHAR(20) NOT NULL
        CHECK (review_type IN ('content', 'calibration', 'fairness', 'technical')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'changes_requested')),
    comments TEXT,
    reviewed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_review_records_item_id ON review_records(item_id);
CREATE INDEX IF NOT EXISTS idx_review_records_reviewer_id ON review_records(reviewer_id);
CREATE INDEX IF NOT EXISTS idx_review_records_status ON review_records(status);

-- Audit log: track all item state changes
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    item_id UUID NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,
    old_value JSONB,
    new_value JSONB,
    performed_by VARCHAR(100),
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_audit_log_item_id ON audit_log(item_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_performed_at ON audit_log(performed_at);

-- Trigger function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to items table
DROP TRIGGER IF EXISTS update_items_updated_at ON items;
CREATE TRIGGER update_items_updated_at
    BEFORE UPDATE ON items
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to domains table
DROP TRIGGER IF EXISTS update_domains_updated_at ON domains;
CREATE TRIGGER update_domains_updated_at
    BEFORE UPDATE ON domains
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default domains
-- Reading & Writing domains (11 domains)
INSERT INTO domains (name, section, category, description, target_percentage) VALUES
('information_and_ideas.central_ideas_and_details', 'reading_writing', 'Information and Ideas', 'Identifying the main idea or primary purpose of a passage', 9.09),
('information_and_ideas.command_of_evidence_textual', 'reading_writing', 'Information and Ideas', 'Selecting evidence that supports a claim', 9.09),
('information_and_ideas.inferences', 'reading_writing', 'Information and Ideas', 'Drawing conclusions from stated information', 9.09),
('information_and_ideas.words_in_context', 'reading_writing', 'Information and Ideas', 'Vocabulary in context, connotation, tone', 9.09),
('craft_and_structure.text_structure_and_purpose', 'reading_writing', 'Craft and Structure', 'How a passage is organized and why', 9.09),
('craft_and_structure.cross_text_connections', 'reading_writing', 'Craft and Structure', 'Synthesis across two related passages', 9.09),
('expression_of_ideas.rhetorical_synthesis', 'reading_writing', 'Expression of Ideas', 'Combining information to achieve a rhetorical goal', 9.09),
('expression_of_ideas.transitions', 'reading_writing', 'Expression of Ideas', 'Logical connectors between sentences and ideas', 9.09),
('standard_english_conventions.boundaries', 'reading_writing', 'Standard English Conventions', 'Sentence structure, run-ons, fragments, punctuation', 9.09),
('standard_english_conventions.form_structure_sense', 'reading_writing', 'Standard English Conventions', 'Subject-verb agreement, pronouns, modifiers', 9.09),
('standard_english_conventions.standard_english', 'reading_writing', 'Standard English Conventions', 'Conventions of usage and style', 9.09)
ON CONFLICT (name) DO NOTHING;

-- Math domains (16 domains)
INSERT INTO domains (name, section, category, description, target_percentage) VALUES
('algebra.linear_equations_one_variable', 'math', 'Algebra', 'Solving and interpreting single-variable equations', 6.25),
('algebra.linear_equations_two_variables', 'math', 'Algebra', 'Systems of equations, intersections', 6.25),
('algebra.linear_functions', 'math', 'Algebra', 'Slope, intercept, rate of change', 6.25),
('algebra.systems_of_linear_equations', 'math', 'Algebra', 'Solving systems of linear equations', 6.25),
('advanced_math.nonlinear_functions', 'math', 'Advanced Math', 'Quadratics, exponentials, absolute value', 6.25),
('advanced_math.nonlinear_equations', 'math', 'Advanced Math', 'Solving quadratic and polynomial equations', 6.25),
('advanced_math.equivalent_expressions', 'math', 'Advanced Math', 'Equivalent expressions and simplifying', 6.25),
('problem_solving_and_data_analysis.ratios_rates_proportions', 'math', 'Problem-Solving and Data Analysis', 'Unit conversion, scaling, proportional reasoning', 6.25),
('problem_solving_and_data_analysis.percentages', 'math', 'Problem-Solving and Data Analysis', 'Percent change, percent of a whole', 6.25),
('problem_solving_and_data_analysis.one_variable_data', 'math', 'Problem-Solving and Data Analysis', 'Mean, median, mode, range, standard deviation', 6.25),
('problem_solving_and_data_analysis.two_variable_data', 'math', 'Problem-Solving and Data Analysis', 'Scatterplots, line of best fit, correlation', 6.25),
('problem_solving_and_data_analysis.probability', 'math', 'Problem-Solving and Data Analysis', 'Simple and conditional probability', 6.25),
('problem_solving_and_data_analysis.inference_from_samples', 'math', 'Problem-Solving and Data Analysis', 'Margin of error, survey design', 6.25),
('geometry_and_trigonometry.area_volume', 'math', 'Geometry and Trigonometry', '2D and 3D geometry', 6.25),
('geometry_and_trigonometry.lines_angles_triangles', 'math', 'Geometry and Trigonometry', 'Geometric relationships and proofs', 6.25),
('geometry_and_trigonometry.right_triangles_trigonometry', 'math', 'Geometry and Trigonometry', 'Pythagorean theorem, SOHCAHTOA, special triangles', 6.25),
('geometry_and_trigonometry.circles', 'math', 'Geometry and Trigonometry', 'Arc length, sector area, equation of a circle', 6.25)
ON CONFLICT (name) DO NOTHING;
